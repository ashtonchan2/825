import pytz
import yfinance as yf    
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

# Obtain S&P 500 tickers as a list
def get_sp500_tickers():
    
    # Obtain S&P500 tickers from Wikipedia page and store in df
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))

    # Return the list of tickers
    tickers = list(df[0].Symbol)
    return tickers

# Getting the history df of ticker using Yfinance
def get_history(ticker, period_start, period_end, granularity="1d", tries=0):

    # Try except catch for calling YFinance on ticker
    # If exception occurs (e.g. Network error, API fails), recursively call get_history
    # If more than 5 tries, return empty df
    try:
        df = yf.Ticker(ticker).history(
            start=period_start, 
            end=period_end,
            interval=granularity,
            auto_adjust=True  # Adjusts OHLC (open, high, low, close) for dividends and stock splits
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries + 1)
        return pd.DataFrame()
    
    # Renaming the columns of the df
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume",
    })

    # If resulting df is empty, return an empty df
    if df.empty:
        return pd.DataFrame()
    
    # Convert the times to UTC for consistency across data
    df["datetime"] = df["datetime"].dt.tz_convert(pytz.utc)
    
    # Set the index to the date / time
    df = df.set_index("datetime", drop=True)
    
    # Drop the dividends and stock splits data
    df = df.drop(columns=["Dividends", "Stock Splits"])
    
    return df

# Obtaining the histories of the tickers using threading
# tickers is a list of tickers
def get_histories(tickers, period_starts, period_ends, granularity="1d"):
    
    # Empty list to store the dfs of ticker history
    dfs = [None] * len(tickers)
    
    # Helper function to obtain history and add to the list
    def _helper(i):

        df = get_history(
            tickers[i], 
            period_starts[i], 
            period_ends[i], 
            granularity=granularity)
        
        dfs[i] = df
    
    # Use threading to speed up multiple API calls to YFinance
    threads = [threading.Thread(target=_helper, args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    
    # Filtering out tickers and dfs with invalid data
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    
    return tickers, dfs

# Creating a dictionary of tickers and their corresponding historical OHLCV data
def get_ticker_dfs(start, end):
    
    from utils import load_pickle, save_pickle
    
    # Check if there already exists the dataset obj
    # Otherwise, obtain the dataset and save it
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")

    except Exception as err:
        tickers = get_sp500_tickers()
        
        starts = [start] * len(tickers)
        ends = [end] * len(tickers)

        tickers, dfs = get_histories(tickers, starts, ends, granularity="1d")
        ticker_dfs = {ticker:df for ticker, df in zip (tickers, dfs)}
        save_pickle("dataset.obj", (tickers, ticker_dfs))
        
    return tickers, ticker_dfs

from utils import Alpha

period_start = datetime(2010,1,1, tzinfo = pytz.utc) 
period_end = datetime.now(pytz.utc)

tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)

testfor = 20
tickers = tickers[:testfor]

from alpha1 import Alpha1
alpha1 = Alpha1(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
df1 = alpha1.run_simulation()
print(df1)