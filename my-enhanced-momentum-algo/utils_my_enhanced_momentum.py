import pandas as pd
from os.path import join as opj
from tqdm import tqdm
from utils_general import *


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


def load_stocks_data(data_path, data_file_name, start_date=None, end_date=None, clean_anomalies_flag=True, group_by_month=True):
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
    #df_prices = df_prices.drop(columns=['^SPX'])

    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    if start_date is not None:
        df_prices = df_prices[df_prices['Date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_prices = df_prices[df_prices['Date'] <= pd.to_datetime(end_date)]

    df_prices.set_index('Date', inplace=True)

    ## resample to monthly data if needed
    if group_by_month:
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
    else:
        # If not grouping by month, return the daily prices directly
        df_prices_daily = df_prices.copy()
        df_prices_daily.index = df_prices_daily.index.strftime('%Y-%m-%d')  # Format index to YYYY-MM-DD
        return df_prices_daily


def load_metadata_per_stock(data_path, metadata_file_name, prices_file_name, start_date=None, end_date=None):
    """
    Load metadata for each stock from a CSV file.

    Parameters:
        data_path (str): Path to the directory containing the metadata file.
        metadata_file_name (str): Name of the CSV file containing stock metadata.

    Returns:
        pd.DataFrame: DataFrame containing stock metadata.
    """
    df_metadata = pd.read_csv(opj(data_path, metadata_file_name))
    # remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_metadata.columns:
        df_metadata = df_metadata.drop(columns=['Unnamed: 0'])
    df_metadata['ticker'] = df_metadata['ticker'].str.strip()  # Clean ticker names
    # set the ticker as the index
    df_metadata.set_index('ticker', inplace=True, drop=False)
    market_cap_per_company = df_metadata['marketCap'].to_dict()  # Save market cap as a dictionary
    # loading the prices data to get the price per stock per day
    df_prices = pd.read_csv(opj(data_path, prices_file_name))

    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    if start_date is not None:
        df_prices = df_prices[df_prices['Date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_prices = df_prices[df_prices['Date'] <= pd.to_datetime(end_date)]

    df_prices.set_index('Date', inplace=True, drop=False)

    df_prices_monthly = df_prices.groupby(pd.Grouper(freq='M')).last()

    # change the index to datetime format (YYY-MM)
    df_prices_monthly.index = df_prices_monthly.index.strftime('%Y-%m')
    # we can now take the current value of each stock, and multiply it by the value the stock was per day
    # Create a copy of the original DataFrame only with stocks that are in the market_cap_dict
    stocks_to_keep = [stock for stock in df_prices_monthly.columns if stock in market_cap_per_company]
    df_prices_monthly_filtered = df_prices_monthly[stocks_to_keep]

    # Multiply each stock's price by its market cap
    for stock in tqdm(stocks_to_keep):
        df_prices_monthly_filtered[stock] = df_prices_monthly_filtered[stock] * market_cap_per_company[stock]

    # converting the df to have 3 columns - date, ticker, and price
    df_prices_monthly_filtered = df_prices_monthly_filtered.reset_index()
    prices_monthly_melted = pd.melt(df_prices_monthly_filtered, id_vars=['Date'], var_name='ticker',
                                    value_name='company_market_cap')
    # Shift the company_market_cap column by one month within each ticker group
    prices_monthly_melted['company_market_cap'] = prices_monthly_melted.groupby('ticker')['company_market_cap'].shift(1)

    # remove cases where price is None
    prices_monthly_melted = prices_monthly_melted.dropna(subset=['company_market_cap'])
    prices_monthly_melted.rename(columns={'Date': 'date'}, inplace=True)
    return df_metadata, prices_monthly_melted


def build_ml_dataset_from_returns(df_returns):
    """
    Given a DataFrame where rows are monthly dates and columns are tickers,
    and values are monthly returns, build a row-per-(date, ticker) ML dataset
    with engineered features and a next-month return target.
    """
    df_returns = df_returns.copy()
    df_returns.index = pd.to_datetime(df_returns.index)
    df_returns = df_returns.sort_index()

    # Generate features
    ret_1m = df_returns.shift(1)
    ret_3m = df_returns.rolling(3).sum().shift(1)
    ret_6m = df_returns.rolling(6).sum().shift(1)
    ret_12m = df_returns.rolling(12).sum().shift(1)

    vol_3m = df_returns.rolling(3).std().shift(1)
    vol_6m = df_returns.rolling(6).std().shift(1)
    vol_12m = df_returns.rolling(12).std().shift(1)

    skew_3m = df_returns.rolling(3).skew().shift(1)

    rolling_max = df_returns.rolling(3).max()
    drawdown_3m = (df_returns - rolling_max) / rolling_max
    drawdown_3m = drawdown_3m.shift(1)

    ma_6m = df_returns.rolling(6).mean()
    ma_12m = df_returns.rolling(12).mean()
    ma_dist_6m = (df_returns - ma_6m) / ma_6m
    ma_dist_12m = (df_returns - ma_12m) / ma_12m

    # Target = next month's return
    target = df_returns.shift(-1)

    # Combine features and target
    feature_dict = {
        'ret_1m': ret_1m,
        'ret_3m': ret_3m,
        'ret_6m': ret_6m,
        'ret_12m': ret_12m,
        'vol_3m': vol_3m,
        'vol_6m': vol_6m,
        'vol_12m': vol_12m,
        'skew_3m': skew_3m,
        'drawdown_3m': drawdown_3m,
        'ma_dist_6m': ma_dist_6m,
        'ma_dist_12m': ma_dist_12m,
        'target': target
    }

    all_rows = []

    for col in df_returns.columns:
        df_feat = pd.DataFrame({
            'date': df_returns.index,
            'ticker': col,
            'ret_1m': feature_dict['ret_1m'][col],
            'ret_3m': feature_dict['ret_3m'][col],
            'ret_6m': feature_dict['ret_6m'][col],
            'ret_12m': feature_dict['ret_12m'][col],
            'vol_3m': feature_dict['vol_3m'][col],
            'vol_6m': feature_dict['vol_6m'][col],
            'vol_12m': feature_dict['vol_12m'][col],
            'skew_3m': feature_dict['skew_3m'][col],
            'drawdown_3m': feature_dict['drawdown_3m'][col],
            'ma_dist_6m': feature_dict['ma_dist_6m'][col],
            'ma_dist_12m': feature_dict['ma_dist_12m'][col],
            'target': feature_dict['target'][col],
        })
        all_rows.append(df_feat)

    df_final = pd.concat(all_rows, ignore_index=True)

    # Drop rows with missing values (early months or last month with no target)
    df_final = df_final.dropna()
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m')
    return df_final


def calc_enhanced_momentum_strategies(df_returns, lookback, holding_period, n_long, n_short,
                                      short_cap, long_cap, predictions_folder):
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
        # extract the date and convert it to datetime
        # prediction_date = df_returns.iloc[i].name.strftime('%Y-%m')  # this is the date of the prediction

        prediction_date = pd.to_datetime(df_returns.iloc[i].name, format='%Y-%m')
        prediction_year = prediction_date.year
        prediction_month = prediction_date.month
        # the next process is the actual chnage of my approach. Here I select the winners and losers based on a model

        winners, losers, winner_weights, loser_weights = (
            rank_and_select_stocks(potential_stocks=list(lookback_returns.columns),
                                   n_long=n_long, n_short=n_short, year=prediction_year, month=prediction_month,
                                   predictions_folder=predictions_folder))

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


def rank_and_select_stocks(potential_stocks, n_long, n_short, year, month, predictions_folder):
    # old logic of extracting the stocks in a random way and uniform weights
    # winners = potential_stocks[0:n_long]
    # losers = potential_stocks[-n_short:]
    # loser_weights = [1 / len(winners) for w in winners]
    # winner_weights = [1 / len(losers) for l in losers]

    # new logic of extracting the stocks based on predictions
    # load the predictions file from disk
    predictions_file = opj(predictions_folder, f'predictions_{year}.csv')
    df_predictions = pd.read_csv(predictions_file)
    # filter the predictions for the potential stocks
    df_predictions_filtered = df_predictions[df_predictions['ticker'].isin(potential_stocks)].copy()
    # filter the predictions for the given year and month
    df_predictions_filtered = df_predictions_filtered[
        (df_predictions_filtered['year'] == year) & (df_predictions_filtered['month'] == month)]
    # sort the predictions by the 'predictions' column in descending order
    df_predictions_filtered = df_predictions_filtered.sort_values(by='predictions', ascending=False)

    # select the top n_long and bottom n_short stocks
    # Extract top n_long and bottom n_short tickers
    winners = df_predictions_filtered.head(n_long)
    losers = df_predictions_filtered.tail(n_short)

    # Calculate weights for top tickers
    winner_weights = list(winners['predictions'] / winners['predictions'].sum())
    # Calculate weights for bottom tickers (negative weights)
    loser_weights = list(losers['predictions'] / losers['predictions'].sum())
    return list(winners['ticker']), list(losers['ticker']), winner_weights, loser_weights
