import lzma
import dill as pickle

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
        units = portfolio_df.loc[idx - 1, "{} units".format(inst)]
        
        # Check if there exists a position in the insturment
        if units != 0:
            # Change in price (current date - prev date)
            delta = dfs[inst].loc[date, "close"] - dfs[inst].loc[prev, "close"] 
            
            # Insturment PNL = delta * units
            inst_pnl = delta * units
            
            # Update our day pnl
            day_pnl += inst_pnl
            
            # Nominal returns = weight * return
            nominal_ret += portfolio_df.loc[idx - 1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    
    # Update capital return, capital
    # Create day PNL, nominal return, and capital return columns
    capital_ret = nominal_ret * portfolio_df.loc[idx - 1, "leverage"]
    portfolio_df.loc[idx, "capital"] = portfolio_df.loc[idx - 1, "capital"] + day_pnl
    portfolio_df.loc[idx, "day_pnl"] = day_pnl
    portfolio_df.loc[idx, "nominal_ret"] = nominal_ret
    portfolio_df.loc[idx, "capital_ret"] = capital_ret
    
    return day_pnl, capital_ret

import pandas as pd
import numpy as np
from datetime import timedelta

class Alpha():

    # insts: insturments (in this case, stock tickers)
    # dfs: historical data of each stock
    # start: start date
    # end: end date
    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = dfs
        self.start = start
        self.end = end
    
    # Creating a "skeleton" portfolio with corresponding trade range
    def init_portfolio_settings(self, trade_range):
        
        # Create a dataframe with the given date range as the index, reset the index, and rename column
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        
        # Create a new column "capital", and set the amount
        portfolio_df.loc[0, "capital"] = 10000
        
        return portfolio_df
     
    def compute_meta_info(self, trade_range):
        
        for inst in self.insts:
            df = pd.DataFrame(index=trade_range)
            
            # Keeping index consistent across all dfs
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            
            # Obtaining daily percentage returns
            # Daily return = (Today price / Yesterday price) - 1
            self.dfs[inst]["ret"] = self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1
            
            # Detects if two consecutive prices are identical
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()

            # Determines whether a inst is recognized as trading based on 5 consecutive identical trading prices
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            
            # Want only elibgible stocks
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
            
        return
    
    def run_simulation(self):
        
        print("running backtest")
        
        # OLD
        # date_range = pd.date_range(start=self.start, end=self.end, freq="D")
        
        # NEW
        start = self.start + timedelta(hours=5)
        end = self.end + timedelta(hours=5)
        date_range = pd.date_range(start, end, freq="D")
        
        self.compute_meta_info(trade_range=date_range)
        
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)

        # Loop through the dates
        for i in portfolio_df.index:
            
            # Particular day
            date = portfolio_df.loc[i, "datetime"]
            
            # Eligible stocks to be traded on a particiular day
            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date, "eligible"]]
            
            # Non-eligible stocks to be traded on a particular day
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]
            
            # Compute daily PNL, capital return
            if i != 0:
                
                # Previous date
                date_prev = portfolio_df.loc[i-1, "datetime"]
                
                day_pnl, capital_ret = get_pnl_stats(
                    date=date, 
                    prev=date_prev, 
                    portfolio_df=portfolio_df, 
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )
            
            alpha_scores = {}
            
            # Compute alpha signals (random for demonstrative purposes)
            import random
            
            for inst in eligibles:
                alpha_scores[inst] = random.uniform(0,1)
            
            # Trade top 25% "high" alpha tickers
            # Short bottom 25% "low" alpha tickers
            alpha_scores = {k:v for k,v in sorted(alpha_scores.items(), key=lambda pair:pair[1])}
            
            # List of top 25% "high" alpha tickers to go long
            alpha_long = list(alpha_scores.keys())[-int(len(eligibles)/4):]
            
            # List of bottom 25% "low" alpha tickers to go short
            alpha_short = list(alpha_scores.keys())[:int(len(eligibles)/4)]
            
            # Ignore trading non eligibles
            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0
            
            # Nominal total of portfolio
            nominal_tot = 0
            
            # Calculating allocation of capital on trades, position, and updating nominal total
            for inst in eligibles:
                
                # Go long if its in the long list, short if in the short list
                forecast = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)
                
                # capital divided equally among number of long and shorted stocks
                dollar_allocation = portfolio_df.loc[i, "capital"] / (len(alpha_long) + len(alpha_short))
                
                # (long or short) * dollar amount / closing price
                position = forecast * dollar_allocation / self.dfs[inst].loc[date, "close"]
                
                # create a new column outline the position: "'ticker' units"
                portfolio_df.loc[i, inst + " units"] = position
                
                # update nominal total with the position amount
                nominal_tot += abs(position * self.dfs[inst].loc[date, "close"])
            
            # Create a weight column for each insturment
            for inst in eligibles:
                
                # Number of units in the position of the insturment
                units = portfolio_df.loc[i, inst + " units"]
                
                # Nominal value of the insturment
                nominal_inst = units * self.dfs[inst].loc[date, "close"]
                
                # Weight of insturment
                inst_w = nominal_inst / nominal_tot
                
                # Create the weight column: "'ticker' w"
                portfolio_df.loc[i, inst + " w"] = inst_w
                
            # Create a nominal total column
            portfolio_df.loc[i, "nominal"] = nominal_tot
            
            # Create a leverage column
            portfolio_df.loc[i, "leverage"] = nominal_tot / portfolio_df.loc[i, "capital"]

            if i % 100 == 0: print(portfolio_df.loc[i])
            # input(portfolio_df.loc[i])
            
        print(alpha_scores)
        print(alpha_long)
        print(alpha_short)
        
        return portfolio_df 

            # compute positions and other information