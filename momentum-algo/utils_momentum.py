import pandas as pd
from tqdm import tqdm
from utils_general import *


def selector(returns, n_long, n_short):
    cumprod_for_sorting = np.prod(1 + returns) - 1
    winners = list(cumprod_for_sorting.nlargest(n_long).keys())
    losers = list(cumprod_for_sorting.nsmallest(n_short).keys())

    # determine weights by cross-sectional scaling
    loser_weights = cumprod_for_sorting[losers].div(cumprod_for_sorting[losers].sum(axis=0))
    winner_weights = cumprod_for_sorting[winners].div(cumprod_for_sorting[winners].sum(axis=0))
    return winners, losers, winner_weights, loser_weights


def clean_anomalies(df, threshold=10, max_anomalies=2):
    """
    Cleans anomalies in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with % changes in stock prices.
        threshold (float): Threshold for identifying anomalies (e.g., 10 for 1000% change).
        max_anomalies (int): Maximum number of anomalies allowed per column before removal.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Identify anomalies
    anomalies = (df.abs() > threshold)

    # Remove columns with too many anomalies
    columns_to_remove = anomalies.sum(axis=0) > max_anomalies
    df_cleaned = df.loc[:, ~columns_to_remove].copy()  # Explicitly create a copy

    # Replace anomalies with 0 in remaining columns
    for col in df_cleaned.columns:
        df_cleaned.loc[anomalies[col], col] = 0

    return df_cleaned


def load_stocks_data(data_path, data_file_name, start_date=None, end_date=None, clean_anomalies_flag=True):
    """
    Load stock data from a CSV file and optionally clean anomalies.

    Parameters:
        data_path (str): Path to the directory containing the data file.
        data_file_name (str): Name of the CSV file containing stock data.
        clean_anomalies_flag (bool): Whether to clean anomalies in the data.

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    df_prices = pd.read_csv(opj(data_path, data_file_name))
    # Remove the ^SPX column from the DataFrame
    if '^SPX' in df_prices.columns:
        df_prices = df_prices.drop(columns=['^SPX'])

    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    if start_date is not None:
        df_prices = df_prices[df_prices['Date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_prices = df_prices[df_prices['Date'] <= pd.to_datetime(end_date)]

    df_prices.set_index('Date', inplace=True)

    ## resample to monthly data
    df_prices_monthly = df_prices.groupby(pd.Grouper(freq='M')).last()

    # change the index to datetime format (YYY-MM)
    df_prices_monthly.index = df_prices_monthly.index.strftime('%Y-%m')
    # calculate ordinary returns from prices as 1-period percentage change. As there is no period before the very first,
    # we better take off the first row, which is full of NaNs.
    df_returns_w_rf = df_prices_monthly.pct_change(1)
    df_returns_w_rf = df_returns_w_rf[1:]  # drop the first row with NaNs

    # clean anomalies in the returns dataframe
    if clean_anomalies_flag:
        df_returns_w_rf_cleaned = clean_anomalies(df_returns_w_rf, threshold=3, max_anomalies=1)
    return df_returns_w_rf_cleaned


def calc_momentum_strategies(df_returns, lookback, holding_period, n_long, n_short, short_cap, long_cap):
    # Weights calculation

    # set up a list of dataframes names for each type of momentum
    l_weights_names = list(['df_weights_pureTS', 'df_weights_TS_cumulative'])  # , 'df_weights_scaling'])
    # set up a list of names of strategies for more intuitive reading, namely for plotting and summary statistics
    strategy_names = list(['Momentum Pure TS', 'Momentum Cumulative TS'])  # , 'Momentum TS & CS'])

    # we need to set up an empty dictionary, where all dataframes will be stored. Option No.1: loop through the names of strategies
    # and write a new dataframe in each iteration of the loop
    dict_weights = {weights_name: pd.DataFrame(data=0, index=df_returns.index, columns=df_returns.keys()) for
                    weights_name in l_weights_names}

    for i in tqdm(range(lookback, len(df_returns))):
        # slicing our returns to desired lookback period, which we then pass to our selector function
        lookback_returns = df_returns.iloc[i - lookback:i, ]

        # using selector function to determine winners and losers in the past "lookback" months.
        # Will be used for weights assingning by subsetting

        winners, losers, winner_weights, loser_weights = selector(lookback_returns, n_long, n_short)

        # we determine the column index of winners/losers to assign weights, because we are looping through integer index
        # using iloc and assignign to a row-wise sliced df and subset by columns. Note it isn't possible to use
        # df.iloc[i+1:i+1:holding_period][winners/loser], because this returns only a COPY of the desired subset and
        # a slice of the original df, but does not access it directly.

        # determine index of winners/loser with our function
        winners_index = column_index(df_returns, winners)
        loser_index = column_index(df_returns, losers)

        # Momentum variation No.1: Pure Momentum. Every period, we determine the winners/losers and assign weights for the
        # following holding period to them based on desired number of titles in long and short leg. Note that this
        # variation makes most sense when holding_period = 1, otherwise we are just overwriting the weights
        dict_weights['df_weights_pureTS'].iloc[i + 1:i + 1 + holding_period, winners_index] = 1 / n_long
        dict_weights['df_weights_pureTS'].iloc[i + 1:i + 1 + holding_period, loser_index] = -1 / n_short

        # Momentum variation No.2: Cumulative. Every period, we determine the winners/loser and assign weights for
        # the following holding period, but add them to the weights which we determined previously.
        # We are cumulating the weights which were determined in previous periods
        dict_weights['df_weights_TS_cumulative'].iloc[i + 1:i + 1 + holding_period, winners_index] = (
                dict_weights['df_weights_TS_cumulative'].iloc[i + 1:i + 1 + holding_period, winners_index] + 1 / n_long)
        dict_weights['df_weights_TS_cumulative'].iloc[i + 1:i + 1 + holding_period, loser_index] = (
                dict_weights['df_weights_TS_cumulative'].iloc[i + 1:i + 1 + holding_period, loser_index] - 1 / n_short)

        # Weights determined - need to scale them and multiply with returns to get returns and then sum them up to get strategy returns
        dict_scaled_weights = {}
        dict_strategy_returns = {}
        dict_portfolio_returns = {}

        # Scaling and returns calculation
        # We need to scale the weights such that the long leg and the short leg sum to 1
        # Also, we take into account long and short caps, meaning we limit how much can we actually go short
        # (it would be possible to have long and short legs sum to 1, while the short leg would have a cumulative weight

        for weights_name, strategy in zip(dict_weights, strategy_names):
            temp_scaled = dict_weights[weights_name].copy()
            temp_not_scaled = dict_weights[weights_name].copy()

            temp_scaled[temp_not_scaled > 0] = (
                    temp_not_scaled.div(temp_not_scaled[temp_not_scaled > 0].sum(axis=1), axis=0) * long_cap)
            temp_scaled[temp_not_scaled < 0] = (
                    (temp_not_scaled.div(temp_not_scaled[temp_not_scaled < 0].sum(axis=1), axis=0)) * short_cap)

            dict_scaled_weights[weights_name + '_scaled'] = temp_scaled
            dict_strategy_returns[strategy] = temp_scaled * df_returns

            # we sum across assets to get portfolio returns, accounting for transactions costs calculate above
            dict_portfolio_returns[strategy] = (temp_scaled * df_returns).sum(
                axis=1)  # - transaction_costs + (df_rf[rf_string] * (1 - own_capital))
        df_all_strategies = pd.DataFrame(dict_portfolio_returns)
    return df_all_strategies
