import pandas as pd
import numpy as np

from utils import get_pnl_stats

# Long-biased momentum strategy
class Alpha3():

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
        
        '''
        ma_faster > ma_slower ? buy : flat (0)
        
        1. fast_crossover 
        2. medium_crossover
        3. slow_crossover 
        
        plus(
            mean_10(close) > mean50(close),
            mean_20(close) > mean100(close),
            mean_50(close) > mean200(close)
        )
        Can result in: 0, 1, 2, 3
        will be our forecast
        i.e. 3 is most confident 
        '''
        
        for inst in self.insts: 
            
            df = pd.DataFrame(index=trade_range)
            
            inst_df = self.dfs[inst]
            
            fast = np.where(inst_df.close.rolling(10).mean() > inst_df.close.rolling(50).mean(), 1, 0)
            medium = np.where(inst_df.close.rolling(20).mean() > inst_df.close.rolling(100).mean(), 1, 0)
            slow = np.where(inst_df.close.rolling(50).mean() > inst_df.close.rolling(200).mean(), 1, 0)
            alpha = fast + medium + slow
            
            self.dfs[inst]["alpha"] = alpha
            
            # Keeping index consistent across all dfs
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            
            # Obtaining daily percentage returns
            # Daily return = (Today price / Yesterday price) - 1
            self.dfs[inst]["ret"] = self.dfs[inst]["close"] / self.dfs[inst]["close"].shift(1) - 1
            
            self.dfs[inst]["alpha"] = self.dfs[inst]["alpha"].ffill()
            
            # Detects if two consecutive prices are identical
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()

            # Determines whether a inst is recognized as trading based on 5 consecutive identical trading prices
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            
            # Want only elibgible stocks
            self.dfs[inst]["eligible"] = eligible.astype(int) \
                & (self.dfs[inst]["close"] > 0).astype(int) \
                & (~pd.isna(self.dfs[inst]["alpha"]))
            
        return
    
    def run_simulation(self):
        
        print("running backtest")
        
        date_range = pd.date_range(start=self.start, end=self.end, freq="D")
        
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
            
            for inst in eligibles:
                alpha_scores[inst] = self.dfs[inst].loc[date, "alpha"]
            
            # Ignore trading non eligibles
            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0
            
            absolute_scores = np.abs([score for score in alpha_scores.values()])
            
            forecast_chips = np.sum(absolute_scores)
            
            # Nominal total of portfolio
            nominal_tot = 0
            
            # Calculating allocation of capital on trades, position, and updating nominal total
            for inst in eligibles:
                
                # Forecast is the direct alpha score, will be negative if not confident, positive if confident
                forecast = alpha_scores[inst]
                
                # capital divided equally among number of stocks
                dollar_allocation = portfolio_df.loc[i, "capital"] / forecast_chips if forecast_chips != 0 else 0
                
                # forecast * dollar amount / closing price
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

            # Display portfolio every 100 days
            if i % 100 == 0: print(portfolio_df.loc[i])
        
        return portfolio_df 
