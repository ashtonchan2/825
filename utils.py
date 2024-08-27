import lzma
import dill as pickle

import time
from functools import wraps

def timeme(func):
    @wraps(func)
    def timeddiff(*args, **kwargs):
        a = time.time()
        result = func(*args, **kwargs)
        b = time.time()
        print(f"@timeme: {func.__name__} took {b - a} seconds")
        return result
    return timeddiff 

def load_pickle(path):
    
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    
    return file        

def save_pickle(path, obj):
    
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)

def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    
    # Daily PNL
    day_pnl = 0
    
    # Nominal returns
    nominal_ret = 0
    
    # Calculate the daily PNL and nominal returns for each insturment
    for inst in insts:
        
        # Number of units in previous day
        units = portfolio_df.at[idx - 1, "{} units".format(inst)]
        
        # Check if there exists a position in the insturment
        if units != 0:
            # Change in price (current date - prev date)
            delta = dfs[inst].at[date, "close"] - dfs[inst].at[prev, "close"] 
            
            # Insturment PNL = delta * units
            inst_pnl = delta * units
            
            # Update our day pnl
            day_pnl += inst_pnl
            
            # Nominal returns = weight * return
            nominal_ret += portfolio_df.at[idx - 1, "{} w".format(inst)] * dfs[inst].at[date, "ret"]
    
    # Update capital return, capital
    # Create day PNL, nominal return, and capital return columns
    capital_ret = nominal_ret * portfolio_df.at[idx - 1, "leverage"]
    portfolio_df.at[idx, "capital"] = portfolio_df.at[idx - 1, "capital"] + day_pnl
    portfolio_df.at[idx, "day_pnl"] = day_pnl
    portfolio_df.at[idx, "nominal_ret"] = nominal_ret
    portfolio_df.at[idx, "capital_ret"] = capital_ret
    
    return day_pnl, capital_ret

import pandas as pd
import numpy as np

from copy import deepcopy

class AbstractImplementationException(Exception):
    pass

class Alpha():

    # insts: insturments (in this case, stock tickers)
    # dfs: historical data of each stock
    # start: start date
    # end: end date
    # portfolio_vol: target portfolio volatility
    def __init__(self, insts, dfs, start, end, portfolio_vol = 0.2):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start
        self.end = end
        self.portfolio_vol = portfolio_vol
    
    # Creating a "skeleton" portfolio with corresponding trade range
    def init_portfolio_settings(self, trade_range):
        
        # Create a dataframe with the given date range as the index, reset the index, and rename column
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        
        # Create a new column "capital", and set the amount
        portfolio_df.at[0, "capital"] = 10000
        portfolio_df.at[0, "day_pnl"] = 0.0
        portfolio_df.at[0, "capital_ret"] = 0.0
        portfolio_df.at[0, "nominal_ret"] = 0.0
        
        return portfolio_df
     
    def pre_compute(self, trade_range):
        pass
     
    def post_compute(self, trade_range):
        pass
    
    def compute_signal_distribution(self, eligible, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")
    
    # @profile
    def compute_meta_info(self, trade_range):
        
        self.pre_compute(trade_range=trade_range)
        
        def is_any_one(x):
            return int(np.any(x)) 
        
        for inst in self.insts:
            df = pd.DataFrame(index=trade_range)
            
            inst_vol = (self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1).rolling(30).std()
            
            # Keeping index consistent across all dfs
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            
            # Obtaining daily percentage returns
            # Daily return = (Today price / Yesterday price) - 1
            self.dfs[inst]["ret"] = self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1
            
            self.dfs[inst]["vol"] = inst_vol
            
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].ffill().fillna(0)
            
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            
            # Detects if two consecutive prices are identical
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()

            # Determines whether an inst is recognized as trading based on 5 consecutive identical trading prices
            eligible = sampled.rolling(5).apply(is_any_one, raw=True).fillna(0)
            
            # Want only elibgible stocks
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
        
        self.post_compute(trade_range=trade_range)
        
        return
    
    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]
    
    @timeme
    def run_simulation(self):
        
        print("RUNNING BACKTEST")
        
        date_range = pd.date_range(start=self.start, end=self.end, freq="D")
        
        self.compute_meta_info(trade_range=date_range)
        
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)
        
        self.ewmas, self.ewstrats = [0.01], [1]
        self.strat_scalars = []
        
        # Loop through the dates
        for i in portfolio_df.index:
            
            # Particular day
            date = portfolio_df.at[i, "datetime"]
            
            # Eligible stocks to be traded on a particiular day
            eligibles = [inst for inst in self.insts if self.dfs[inst].at[date, "eligible"]]
            
            # Non-eligible stocks to be traded on a particular day
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]
            
            strat_scalar = 2
            
            # Compute daily PNL, capital return
            if i != 0:
                
                # Previous date
                date_prev = portfolio_df.at[i-1, "datetime"]
                
                strat_scalar = self.get_strat_scaler(
                    target_vol = self.portfolio_vol,
                    ewmas = self.ewmas,
                    ewstrats = self.ewstrats
                )
                
                day_pnl, capital_ret = get_pnl_stats(
                    date=date, 
                    prev=date_prev, 
                    portfolio_df=portfolio_df, 
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )
                
                self.ewmas.append(0.06 * (capital_ret ** 2) + 0.94 * self.ewmas[-1] if capital_ret != 0 else self.ewmas[-1])
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret != 0 else self.ewstrats[-1])
                
            self.strat_scalars.append(strat_scalar)
            
            forecasts, forecast_chips = self.compute_signal_distribution(eligibles, date)
            
            # Ignore trading non eligibles
            for inst in non_eligibles:
                portfolio_df.at[i, "{} w".format(inst)] = 0
                portfolio_df.at[i, "{} units".format(inst)] = 0
            
            # Target volatility sizing for each insturment
            vol_target = (self.portfolio_vol / np.sqrt(253)) * portfolio_df.at[i, "capital"]
            
            # Nominal total of portfolio
            nominal_tot = 0
            
            # Calculating allocation of capital on trades, position, and updating nominal total
            for inst in eligibles:
                
                forecast = forecasts[inst]
                
                scaled_forecast = forecast / forecast_chips if forecast_chips != 0 else 0
                
                position = \
                    scaled_forecast \
                    * strat_scalar \
                    * vol_target \
                    / (self.dfs[inst].at[date, "vol"] * self.dfs[inst].at[date, "close"])
                
                # create a new column outline the position: "'ticker' units"
                portfolio_df.at[i, inst + " units"] = position
                
                # update nominal total with the position amount
                nominal_tot += abs(position * self.dfs[inst].at[date, "close"])
            
            # Create a weight column for each insturment
            for inst in eligibles:
                
                # Number of units in the position of the insturment
                units = portfolio_df.at[i, inst + " units"]
                
                # Nominal value of the insturment
                nominal_inst = units * self.dfs[inst].at[date, "close"]
                
                # Weight of insturment
                inst_w = nominal_inst / nominal_tot
                
                # Create the weight column: "'ticker' w"
                portfolio_df.at[i, inst + " w"] = inst_w
                
            # Create a nominal total column
            portfolio_df.at[i, "nominal"] = nominal_tot
            
            # Create a leverage column
            portfolio_df.at[i, "leverage"] = nominal_tot / portfolio_df.at[i, "capital"]
        
        return portfolio_df.set_index("datetime", drop=True)

from collections import defaultdict
class Portfolio(Alpha):

    def __init__(self, insts, dfs, start, end, stratdfs):
        super().__init__(insts, dfs, start, end)
        self.stratdfs=stratdfs
     
    def post_compute(self, trade_range):
        
        self.positions = {}
        
        for inst in self.insts:
            
            inst_weights = pd.DataFrame(index=trade_range)
            
            for i in range(len(self.stratdfs)):
                inst_weights[i] = self.stratdfs[i]["{} w".format(inst)]\
                    * self.stratdfs[i]["leverage"]
                
                inst_weights[i] = inst_weights[i].ffill().fillna(0.0)
            
            self.positions[inst] = inst_weights
        
    
    def compute_signal_distribution(self, eligibles, date):
        
        forecasts = defaultdict(float)

        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                # Parity risk allocation
                forecasts[inst] += self.positions[inst].at[date, i] * (1 / len(self.stratdfs))
        
        forecast_chips = np.sum(np.abs(list(forecasts.values())))
        
        return forecasts, forecast_chips