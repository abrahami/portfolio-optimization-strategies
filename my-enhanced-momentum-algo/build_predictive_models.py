import sys
sys.path.append("/home/isabrah/etoro/protfolio-optimization")
sys.path.append("/home/isabrah/etoro/protfolio-optimization/libs")
from utils_my_enhanced_momentum import *
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# Set pandas to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


# Configrations
# define start date for Yahoo import
start_date = '2004-01-01'
end_date = '2025-01-01'
download_data = False  # if True, we download the data from Yahoo Finance, otherwise we load it from a pickle file
data_path = '/home/isabrah/etoro/data'
prices_file_name = 'df_prices_clean.csv'#'leading_indices_prices_clean.csv'#'df_prices_clean.csv' #df_prices_clean.csv
metadata_file_name = 'companies_info_as_df.csv'
saving_folder = '/home/isabrah/etoro/predictions'
model = Ridge(alpha=1.0)#GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42) #

if __name__ == '__main__':
    # data loading. I assume this is an already downloaded data, but if not, we can use yfinance to download it.
    df_returns = load_stocks_data(data_path, prices_file_name, start_date=start_date, end_date=end_date,
                                  clean_anomalies_flag=True, group_by_month=True)

    # taking only the first 1000 columns (companies) for testing purposes
    #df_returns = df_returns.iloc[:, :1000].copy()
    print(df_returns.shape)

    # creating the modeling object
    modeling_df = build_ml_dataset_from_returns(df_returns=df_returns)

    print(modeling_df.shape)
    # number of stocks possible to trade with
    n_stocks = modeling_df.shape[0]

    df_stocks_metadata, prices_monthly_melted = load_metadata_per_stock(data_path=data_path, metadata_file_name=metadata_file_name,
                                                                        prices_file_name=prices_file_name)

    prices_monthly_melted.rename(columns={'Date': 'date'}, inplace=True)
    # adding the new feature I created to the df (simple join)
    modeling_df_enriched = modeling_df.merge(prices_monthly_melted,
                                             on=['ticker', 'date'],  # Columns to join on
                                             how='inner')
    # define target column for the model
    target_col = 'target'
    # # using a single column from the metadata (for now) to try plug it into the model
    feature_to_use = ['marketCap']
    df_stocks_metadata_subset = df_stocks_metadata[feature_to_use].copy()

    # # join the modeling DataFrame with the metadata subset df
    modeling_df_enriched = modeling_df_enriched.join(df_stocks_metadata_subset, on='ticker', how='inner')
    # preprocessing (null removal etc)
    df_clean = modeling_df_enriched.dropna()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    # Replace infinite values with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    # remove rows with missing values
    df_clean = df_clean.dropna()

    # adding the year and month columns
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month

    # looping over each year in the dataset, and per each we build a prediction model
    years_in_dataset = list(df_clean['year'].unique())
    for cur_max_year in tqdm(years_in_dataset):
        print(f"Processing year: {cur_max_year}", flush=True)

        # filter the data for the current year
        df_up_to_year = df_clean[df_clean['year'] < cur_max_year].copy()
        if df_up_to_year.shape[0] == 0:
            continue

        df_cur_year_only = df_clean[df_clean['year'] == cur_max_year].copy()
        explanatory_features_for_train = df_up_to_year.drop(columns=['date', 'ticker', target_col, 'year', 'month']).copy()
        explanatory_features_for_test = df_cur_year_only.drop(columns=['date', 'ticker', target_col, 'year', 'month']).copy()
        explanatory_features_for_train = explanatory_features_for_train.clip(lower=-1e10, upper=1e10)
        explanatory_features_for_test = explanatory_features_for_test.clip(lower=-1e10, upper=1e10)
        y_train = df_up_to_year[target_col].copy()
        y_test = df_cur_year_only[target_col].copy()

        # build the model
        model.fit(explanatory_features_for_train, y_train)
        y_pred_train = model.predict(explanatory_features_for_train)
        r2_train = r2_score(y_train, y_pred_train)
        y_pred_test = model.predict(explanatory_features_for_test)
        r2_test = r2_score(y_test, y_pred_test)
        print(f'Current year: {cur_max_year}, Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}', flush=True)

        # adding metadata (year, month and ticker) to the predictions and saving it
        df_predictions = df_cur_year_only[['year', 'month', 'ticker']].copy()
        df_predictions['predictions'] = y_pred_test
        saving_file_name = f'predictions_{cur_max_year}.csv'
        df_predictions.to_csv(opj(saving_folder, saving_file_name), index=False)
