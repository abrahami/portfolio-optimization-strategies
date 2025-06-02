# Source code is taken from here: https://www.financeclub.ch/blog/momentum-strategy---an-introduction-to-quantitative-trading
import sys
sys.path.append("/home/isabrah/etoro/protfolio-optimization")
sys.path.append("/home/isabrah/etoro/protfolio-optimization/libs")
from utils_my_enhanced_momentum import *
from portfolio_analysis import PortfolioAnalysis


# Set pandas to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


# Configrations
# define start date for Yahoo import
start_date = '2010-01-01'
end_date = '2025-01-01'
download_data = False  # if True, we download the data from Yahoo Finance, otherwise we load it from a pickle file
data_path = '/home/isabrah/etoro/data'
prices_file_name = 'df_prices_clean.csv'#'leading_indices_prices_clean.csv' #df_prices_clean.csv
predictions_folder = '/home/isabrah/etoro/predictions'
results_path = '/home/isabrah/etoro/results'

# define lookback period, i.e. how much back in time do we look to determine which stocks performed the best and worst
# etotro uses 12 months in their instructions file
lookback = 12
# define holding period, i.e. how much into the future do we think the returns will be persisent
# etotro uses 3 months in their instructions file
holding_period = 3

own_capital = 0#1 # how much of our own capital we are using in decimal (1 = 100%). etoro asked to use zero and make long-short portfolio balanced

# define short_cap used for scaling final weights in the short leg of the strategy
short_cap = -1

# define long_cap for scaling final weights in the long leg. We want the legs to sum up to 1 (defined above),
# and the short leg finances the rest of the long long leg. We manipulate equation " 1 = short_cap + long_cap "
long_cap = own_capital - short_cap
plt_saving_name = f'enhanced_momentum_NAVs_{start_date}_to_{end_date}_own_capital_{own_capital}.png'

if __name__ == '__main__':
    # data loading. I assume this is an already downloaded data, but if not, we can use yfinance to download it.
    df_returns = load_stocks_data(data_path, prices_file_name, start_date=start_date, end_date=end_date,
                                  clean_anomalies_flag=True)

    #df_returns = df_returns.iloc[:, :100].copy()  # taking only the first 500 columns (companies)
    # number of stocks possible to trade with
    n_stocks = len(df_returns.columns)

    # define number of stocks in the long and short leg
    n_long = max(int(n_stocks/10), 3)  # at least 3 stocks in the long leg, but not more than 10% of all stocks
    n_short = max(int(n_stocks/10), 3)  # same for the short leg

    # taking only X years (just for debugging purposes, otherwise we would take all the data))
    #df_returns = df_returns.iloc[-12*4:, ]  # taking only last XX years of data, i.e. 24 months

    df_all_strategies = calc_enhanced_momentum_strategies(df_returns, lookback, holding_period, n_long,
                                                          n_short, short_cap, long_cap, predictions_folder)
    # making a benchmark which takes 50% of the stocks in short and the rest in long
    # Randomly select half of the stocks for long and short positions
    n_stocks = len(df_returns.columns)
    long_stocks = np.random.choice(df_returns.columns, size=n_stocks // 2, replace=False)
    short_stocks = df_returns.columns.difference(long_stocks)

    # Calculate the average returns for long and short positions
    if own_capital >= 1:
        df_BM = pd.DataFrame(df_returns.mean(axis=1))
        df_BM.columns = ['Benchmark']
    else:
        df_BM = pd.DataFrame({
            'Long': df_returns[long_stocks].mean(axis=1),
            'Short': df_returns[short_stocks].mean(axis=1)
        })
        df_BM['Benchmark'] = (df_BM['Long'] - df_BM['Short']) / 2
        df_BM.drop(['Long', 'Short'], axis=1, inplace=True)  # drop the intermediate columns
    summary = PortfolioAnalysis(returns=df_all_strategies, ann_factor=12, benchmark=df_BM, rf=None)
    df_summary = summary.analysis_with_benchmark()
    print(df_summary)
    # saving the summary of the analysis
    summary_file_name = f'enhanced_momentum_own_capital_{own_capital}_start_date_{start_date}.csv'
    df_summary.to_csv(opj(results_path, summary_file_name), index=True)

    df_NAVs = summary.net_asse_values  # saving a df of the NAVs we calculated in the class "port_an"
    df_NAVs['Date'] = pd.to_datetime(df_NAVs.index)
    # saving the figure of NAVs
    figure_saving_file_name = f'enhanced_momentum_NAVs_own_capital_{own_capital}_start_date{start_date}.png'
    plot_nav_values(df_NAVs, results_path, labels=['Enhanced Momentum Algorithm', 'Enhanced Cumulative Momentum Algorithm', 'Benchmark (Random Portfolio)'],
                    saving_file_name=figure_saving_file_name)
    print("code has ended successfully. figures are saved in the data folder.", flush=True)

