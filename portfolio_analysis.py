import pandas as pd
import numpy as np
import statsmodels.api as sm


class PortfolioAnalysis:
    def __init__(self, returns, ann_factor, benchmark=None, rf=None):
        self.returns = returns
        self.ann_factor = ann_factor

        if rf is None:
            rf = pd.DataFrame(data=0, index=returns.index, columns=['rf'])

        else:
            rf.columns = ['rf']
        if benchmark is None:
            returns_all = returns

        else:
            self.benchmark = benchmark.sub(rf['rf'], axis=0)
            returns_all = pd.concat([returns, benchmark], axis=1)

        xs_returns = returns_all
        self.xs_returns = xs_returns
        n_periods = len(returns_all)
        self.n_periods = n_periods

        # Calculate excess returns, which is multiplications of each period by (1 + rf)
        net_asse_values = (1 + returns_all).cumprod()
        self.net_asse_values = net_asse_values

        # the first item is just the total return at the end of the period, this is taken by the power of
        geo_avg = (1 + returns_all).prod() ** (ann_factor / n_periods) - 1
        self.geo_avg = geo_avg
        geo_avg_xs = (1 + xs_returns).prod() ** (ann_factor / n_periods) - 1
        self.geo_avg_xs = geo_avg_xs

        vol = xs_returns.std() * np.sqrt(ann_factor)
        SR = geo_avg_xs / vol

        self.SR = SR
        self.vol = vol
        self.skew_strat = returns_all.skew()
        self.kurtosis_strat = returns_all.kurtosis()
        self.alphas = list()
        self.betas = list()
        if benchmark is None:
            benchmark = 0
        else:
            for strat in xs_returns.columns:
                try:
                    OLS_model = sm.OLS(xs_returns[strat], sm.add_constant(self.benchmark)).fit()
                    self.alphas.append(OLS_model.params[0])
                    self.betas.append(OLS_model.params[1])
                except Exception as e:
                    print(f"Error in regression for {strat}: {e}")
                    self.alphas.append(0)
                    self.betas.append(0)
        # calculating the Max Drawdown
        max_drawdown = list()
        for strat in xs_returns.columns:
            cur_drawdown = self.calculate_max_drawdown(returns=xs_returns[strat])
            max_drawdown.append(round(cur_drawdown, 4))
        self.max_drawdown = max_drawdown

    def analysis_with_benchmark(self):
        self.IR = (self.geo_avg_xs.iloc[0:-1] - self.geo_avg_xs.iloc[-1]) / self.vol
        df_summary = pd.DataFrame(index=self.xs_returns.keys(),
                                  columns=['geo_mean_in_%', 'volatility_annual_in_%', 'sharpe_ratio',
                                           'information_ratio', 'skewness', 'excess_kurtosis', 'beta', 'alpha_in_%',
                                           'max_drawdown'])
        df_summary.iloc[:, 0] = round(self.geo_avg_xs * 100, 2)
        df_summary.iloc[:, 1] = round(self.vol * 100, 2)
        df_summary.iloc[:, 2] = round(self.SR, 2)
        df_summary.iloc[:, 3] = round(self.IR, 2)
        df_summary.iloc[:, 4] = round(self.skew_strat, 2)
        df_summary.iloc[:, 5] = round(self.kurtosis_strat, 2)
        df_summary.iloc[:, 6] = self.betas
        df_summary.iloc[:, 6] = round(df_summary.iloc[:, 6], 2)
        df_summary.iloc[:, 7] = self.alphas
        df_summary.iloc[:, 7] = round(df_summary.iloc[:, 7]*100, 2)
        df_summary.iloc[:, 8] = self.max_drawdown
        return df_summary.fillna(0)

    def analysis_without_benchmark(self):
        df_summary = pd.DataFrame(index=self.xs_returns.keys(),
                                  columns=['geo_mean_in_%', 'volatility_annual_in_%', 'sharpe_ratio',
                                           'information_ratio', 'skewness', 'excess_kurtosis', 'beta', 'alpha_in_%',
                                           'max_drawdown'])
        df_summary.iloc[:, 0] = round(self.geo_avg_xs * 100, 2)
        df_summary.iloc[:, 1] = round(self.vol * 100, 2)
        df_summary.iloc[:, 2] = round(self.SR, 2)
        df_summary.iloc[:, 3] = 0
        df_summary.iloc[:, 4] = round(self.skew_strat, 2)
        df_summary.iloc[:, 5] = round(self.kurtosis_strat, 2)
        df_summary.iloc[:, 6] = 0
        df_summary.iloc[:, 7] = 0
        df_summary.iloc[:, 8] = round(self.max_drawdown, 2)
        return df_summary.fillna(0)

    @staticmethod
    def calculate_max_drawdown(returns):
        """
        Calculate the maximum drawdown of a portfolio.
        """

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative_returns.cummax()

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # Find the maximum drawdown
        max_drawdown = drawdown.min()
        return max_drawdown
