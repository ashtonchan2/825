import numpy as np
import pandas as pd

from utils import get_pnl_stats
class Alpha1():
    
    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = dfs
        self.start = start 
        self.end = end

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        portfolio_df.loc[0,"capital"] = 10000
        return portfolio_df

    def compute_meta_info(self,trade_range):
        '''
        https://hangukquant.substack.com/p/formulaic-alphas-5ae
        mean_12(
            neg(
                cszscre(
                    mult(
                        volume,
                        div(
                            minus(minus(close,low),minus(high,close)),
                            minus(high,low)
                        )
                    )
                )
            )
        )
        '''
        op4s = []
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low ) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3       
            
            self.dfs[inst] = df.join(self.dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = -1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
            self.dfs[inst]["op4"] = op4
            op4s.append(self.dfs[inst]["op4"])
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
        
        temp_df = pd.concat(op4s,axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0)
        zscore = lambda x: (x - np.mean(x))/np.std(x)
        cszcre_df = temp_df.fillna(method="ffill").apply(zscore, axis=1)
        for inst in self.insts:
            self.dfs[inst]["alpha"] = cszcre_df[inst].rolling(12).mean() * -1
            self.dfs[inst]["eligible"] = self.dfs[inst]["eligible"] & (~pd.isna(self.dfs[inst]["alpha"]))
        return 

    def run_simulation(self):
        print("running backtest")
        date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)
        for i in portfolio_df.index:
            date = portfolio_df.loc[i,"datetime"]

            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date,"eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]

            if i != 0:
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
            alpha_scores = {k:v for k,v in sorted(alpha_scores.items(),key=lambda pair:pair[1])}
            alpha_long = list(alpha_scores.keys())[-int(len(eligibles)/4):]
            alpha_short = list(alpha_scores.keys())[:int(len(eligibles)/4)]
            
            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0
            
            nominal_tot = 0
            for inst in eligibles:
                forecast = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)
                dollar_allocation = portfolio_df.loc[i,"capital"] / (len(alpha_long) + len(alpha_short))
                position = forecast * dollar_allocation / self.dfs[inst].loc[date,"close"]
                portfolio_df.loc[i, inst + " units"] = position 
                nominal_tot += abs(position * self.dfs[inst].loc[date,"close"])

            for inst in eligibles:
                units = portfolio_df.loc[i, inst + " units"]
                nominal_inst = units * self.dfs[inst].loc[date,"close"]
                inst_w = nominal_inst / nominal_tot
                portfolio_df.loc[i, inst + " w"] = inst_w
            
            portfolio_df.loc[i, "nominal"] = nominal_tot
            portfolio_df.loc[i, "leverage"] = nominal_tot / portfolio_df.loc[i, "capital"]
            if i%100 == 0: print(portfolio_df.loc[i])
        return portfolio_df