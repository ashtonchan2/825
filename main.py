import pytz
import yfinance as yf    
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from utils import timeme
from utils import save_pickle, load_pickle
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3
from utils import Portfolio


# Obtain S&P 500 tickers as a list
def get_sp500_tickers():
    
    # Scrape S&P500 tickers from Wikipedia page and store in df
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
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
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    # Renaming the columns of the df
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })

    # If resulting df is empty, return an empty df
    if df.empty:
        return pd.DataFrame()
    
    # Convert the times to UTC for consistency across data
    df.datetime = pd.DatetimeIndex(df.datetime.dt.date).tz_localize(pytz.utc)
    
    # Drop the dividends and stock splits data
    df = df.drop(columns=["Dividends", "Stock Splits"])
    
    # Set the index to the date / time
    df = df.set_index("datetime", drop=True)
    
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

# Creating a dictionary of tickers and their corresponding historical OHLCV data, caching
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

def main():
    period_start = datetime(2010,1,1, tzinfo = pytz.utc) 
    period_end = datetime(2023,8,31, tzinfo = pytz.utc)

    # CURRENT DATE
    # period_end = datetime.now(pytz.utc)

    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)

    testfor = 20
    print(f"testing {testfor} out of {len(tickers)} tickers")
    tickers = tickers[:testfor]

    alpha1 = Alpha1(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha2 = Alpha2(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha3 = Alpha3(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)

    df1 = alpha1.run_simulation()
    print(list(df1.capital)[-1])
    
    df2 = alpha2.run_simulation()
    print(list(df2.capital)[-1])
    
    df3 = alpha3.run_simulation()
    print(list(df3.capital)[-1])

if __name__ == "__main__":
    main()
    
'''
> Initial runtime
testing 20 out of 501 tickers
RUNNING BACKTEST
@timeme: run_simulation took 56.50452995300293 seconds
30808.013664769336

         179132928 function calls (176956626 primitive calls) in 58.956 seconds

   Ordered by: cumulative time
   List reduced from 4365 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    626/1    0.003    0.000   58.956   58.956 {built-in method builtins.exec}
        1    0.027    0.027   58.956   58.956 main.py:1(<module>)
        1    0.000    0.000   58.458   58.458 main.py:123(main)
        1    0.000    0.000   56.505   56.505 utils.py:8(timeddiff)
        1    0.513    0.513   56.505   56.505 utils.py:153(run_simulation)
   229586    0.833    0.000   32.705    0.000 indexing.py:882(__setitem__)
229632/229586    0.729    0.000   27.848    0.000 indexing.py:1785(_setitem_with_indexer)
   229586    0.466    0.000   25.907    0.000 indexing.py:1946(_setitem_with_indexer_split_path)
   229586    0.750    0.000   24.734    0.000 indexing.py:2111(_setitem_single_column)
  1284201    1.998    0.000   22.259    0.000 indexing.py:1176(__getitem__)
   229587    0.269    0.000   15.748    0.000 generic.py:6432(dtypes)
   954875    0.948    0.000    9.604    0.000 frame.py:4191(_get_value)
     4990    0.172    0.000    8.773    0.002 utils.py:29(get_pnl_stats)
240052/240031    1.188    0.000    8.629    0.000 series.py:389(__init__)
   229587    0.236    0.000    7.083    0.000 managers.py:287(get_dtypes)
   230042    5.937    0.000    5.937    0.000 {built-in method numpy.array}
   229586    0.350    0.000    5.551    0.000 managers.py:1298(column_setitem)
        1    0.001    0.001    4.973    4.973 utils.py:114(compute_meta_info)
   955297    0.720    0.000    4.417    0.000 frame.py:4626(_get_item_cache)
39680312/39654573    2.585    0.000    4.104    0.000 {built-in method builtins.isinstance}


> Changed .loc to .at
testing 20 out of 501 tickers
RUNNING BACKTEST
@timeme: run_simulation took 20.490746021270752 seconds
30808.013664769336

         64989659 function calls (64879222 primitive calls) in 21.578 seconds

   Ordered by: cumulative time
   List reduced from 4385 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    626/1    0.003    0.000   21.578   21.578 {built-in method builtins.exec}
        1    0.028    0.028   21.578   21.578 main.py:1(<module>)
        1    0.000    0.000   21.022   21.022 main.py:123(main)
        1    0.000    0.000   18.968   18.968 utils.py:8(timeddiff)
        1    0.369    0.369   18.968   18.968 utils.py:153(run_simulation)
   954875    0.567    0.000   10.140    0.000 indexing.py:2568(__getitem__)
   954875    0.553    0.000    9.151    0.000 indexing.py:2518(__getitem__)
   954875    0.847    0.000    8.457    0.000 frame.py:4191(_get_value)
        1    0.001    0.001    4.902    4.902 utils.py:114(compute_meta_info)
     4990    0.150    0.000    4.102    0.001 utils.py:29(get_pnl_stats)
   955297    0.617    0.000    3.921    0.000 frame.py:4626(_get_item_cache)
       60    0.000    0.000    3.901    0.065 rolling.py:562(_apply)
       60    0.000    0.000    3.901    0.065 rolling.py:460(_apply_columnwise)
       60    0.000    0.000    3.900    0.065 rolling.py:440(_apply_series)
       60    0.000    0.000    3.897    0.065 rolling.py:595(homogeneous_func)
       60    0.001    0.000    3.897    0.065 rolling.py:601(calc)
       20    0.000    0.000    3.895    0.195 rolling.py:2016(apply)
       20    0.000    0.000    3.895    0.195 rolling.py:1471(apply)
       20    0.091    0.005    3.893    0.195 rolling.py:1531(apply_func)
   690322    0.779    0.000    3.085    0.000 datetimes.py:582(get_loc)
   
   
Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 3.81893 s
File: /Users/ashtonchan/Documents/PROJECT/825/utils.py
Function: compute_meta_info at line 114

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   114                                               @profile
   115                                               def compute_meta_info(self, trade_range):
   116                                                   
   117         1       8913.0   8913.0      0.2          self.pre_compute(trade_range=trade_range)
   118                                                   
   119        21          7.0      0.3      0.0          for inst in self.insts:
   120        20       3467.0    173.3      0.1              df = pd.DataFrame(index=trade_range)
   121                                                       
   122        20       7892.0    394.6      0.2              inst_vol = (self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1).rolling(30).std()
   123                                                       
   124                                                       # Keeping index consistent across all dfs
   125        20      17966.0    898.3      0.5              self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
   126                                                       
   127                                                       # Obtaining daily percentage returns
   128                                                       # Daily return = (Today price / Yesterday price) - 1
   129        20       6127.0    306.4      0.2              self.dfs[inst]["ret"] = self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1
   130                                                       
   131        20       6942.0    347.1      0.2              self.dfs[inst]["vol"] = inst_vol
   132                                                       
   133        20       3313.0    165.7      0.1              self.dfs[inst]["vol"] = self.dfs[inst]["vol"].ffill().fillna(0)
   134                                                       
   135        20       2690.0    134.5      0.1              self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
   136                                                       
   137                                                       # Detects if two consecutive prices are identical
   138        20       3044.0    152.2      0.1              sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()
   139                                           
   140                                                       # Determines whether an inst is recognized as trading based on 5 consecutive identical trading prices
   141        20    3066788.0 153339.4     80.3              eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
   142                                                       
   143                                                       # Want only elibgible stocks
   144        20       7363.0    368.1      0.2              self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
   145                                                   
   146         1     684415.0 684415.0     17.9          self.post_compute(trade_range=trade_range)
   147                                                   
   148         1          0.0      0.0      0.0          return
   
   
> Modified "compute_meta_info": .apply(raw=False) -> .apply(raw=True)
testing 20 out of 501 tickers
RUNNING BACKTEST
@timeme: run_simulation took 5.152562141418457 seconds
30808.013664769336
'''