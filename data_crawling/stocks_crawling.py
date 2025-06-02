
import yfinance as yf
import json

# Configurations
# define start date for Yahoo import. the excerice explicitly mention that we need data from 2004 to 2024
start_date = '2004-01-01'

if __name__ == '__main__':
    # get list of tickers, the list is provided here: https://www.sec.gov/file/company-tickers
    # I have saved it and then load it
    # Load the JSON file
    with open("company_tickers.json", "r") as f:
        data = json.load(f)

    # Extract only the ticker symbols
    tickers = [entry["ticker"] for entry in data.values()]

    print(tickers[:10])  # Preview the first 10 tickers
    print(len(tickers))  # ~10K tickers

    # Download the data per ticker
    df_prices = yf.download(tickers, start=start_date)['Close']
    # clean the df to get only stocks with full information from 2004 to 2024
    df_prices_clean = df_prices.dropna(axis=1)
    df_prices_clean = df_prices_clean.loc[:, ~df_prices.isna().all()]
    df_prices_clean.to_csv('df_prices_clean.csv')