
import yfinance as yf
import json
import time
from tqdm import tqdm

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
    data = []
    max_retries = 3
    initial_sleep = 10
    # fields to download
    fields = [
        'longName', 'sector', 'industry', 'country', 'fullTimeEmployees',
        'marketCap', 'sharesOutstanding', 'floatShares',
        'heldPercentInsiders', 'heldPercentInstitutions',
        'forwardPE', 'trailingPE', 'priceToBook', 'enterpriseToRevenue', 'enterpriseToEbitda',
        'revenueGrowth', 'earningsGrowth', 'ebitdaMargins', 'profitMargins',
        'grossMargins', 'returnOnAssets', 'returnOnEquity',
        'dividendYield', 'dividendRate', 'payoutRatio', 'averageVolume'
    ]

    for i, ticker in tqdm(enumerate(tickers)):
        retry_count = 0
        sleep_time = initial_sleep
        while retry_count <= max_retries:
            try:
                info = yf.Ticker(ticker).info
                record = {'ticker': ticker}
                for f in fields:
                    record[f] = info.get(f, None)
                data.append(record)
                break  # success, break out of retry loop
            except Exception as e:
                if "Too Many Requests" in str(e):
                    retry_count += 1
                    print(f"Rate limited. Sleeping for {sleep_time} seconds (attempt {retry_count})...")
                    time.sleep(sleep_time)
                    sleep_time *= 3  # exponential backoff
                else:
                    print(f"Error fetching {ticker}: {e}")
                    break  # fail for other reasons

