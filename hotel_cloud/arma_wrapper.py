# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class arma_wrapper:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forecast_len: int, # length that need to forecast
                ):
        
        if isinstance(arr, pd.Series):  # type conversion 
            self.arr = arr.to_numpy()
        else:
            self.arr = arr

        if self.arr[-1] < self.arr[0]:  # flip arr to ascent order 
            self.arr = np.flip(self.arr)

        n = len(self.arr)
        if n > history:  #  truncate history 
            self.arr = self.arr[n-history:]
            n=len(self.arr)

        if forecast_len >= n:  # check forecast_len
            raise ValueError("history too short")

        self.forecast_len = forecast_len
        self.fit_arr = self.arr[:int(n-forecast_len)]
        self.model = None
        self.res = None

    def auto_fit(self, 
                order_range: tuple,  # (p, q) to search for
                metric: str = "aic",  # metric to score model; "aic" or "bic"
                verbose=True,  
                ):

        p, q = order_range[0], order_range[1]

        scores = np.zeros((p, q))

        for i in range(p):
            for j in range(q):

                if i==0 and j==0:
                    scores[i][j]=np.inf
                    continue 

                try:
                    mod = sm.tsa.statespace.SARIMAX(self.fit_arr, order=(i,0,j), enforce_invertibility=False)
                    res = mod.fit(disp=False)
                    if metric == "aic":
                        scores[i][j] = res.aic
                    elif metric == "bic":
                        scores[i][j] = res.bic 
                except:
                    scores[i][j] = np.inf

        p, q = np.where(scores==scores.min()); p, q = int(p), int(q)
        if verbose:
            print("scores matrix:")
            print(scores)
            print("")
            print(f"optimal (p, q) is {p, q}")
        
        self.model = sm.tsa.statespace.SARIMAX(self.fit_arr, order=(p, 0, q))
        self.res = self.model.fit()
        return self
    
    @property
    def forecast(self):
        return self.res.forecast(steps=self.forecast_len) if self.res is not None else None

    def plot_forecast(self, stay_date="0"):
        pred = self.forecast; n = len(self.arr)

        if pred is None:
            print("haven't fit model")
            return None

        fig, ax = plt.subplots()

        ax.plot([i for i in range(n)], np.flip(self.arr), color="black", label = "time series")

        # temp = [n - self.forecast_len + i for i in range(self.forecast_len)]
        ax.plot([i for i in range(self.forecast_len)], np.flip(pred), color="red", label="forecasting")

        ax.axvline(x = self.forecast_len-1)

        ax.set_xlim(n, 0); ax.set_xlabel('days before'); ax.set_ylabel('bookings'); ax.grid(True); ax.set_title(f"stay date: {stay_date}")
        plt.show()

        return None


    @property
    def stats(self,):
        print("total history: ", len(self.arr))
        print("data len: ", len(self.fit_arr))
        print("forecast len: ", self.forecast_len)

        if self.model is not None:
            p, q = self.model.model_orders["ar"], self.model.model_orders["ma"]
            print(f"model order (p, q): ({p, q})")

        if self.res is not None:
            pass # TODO add stats


if __name__ == "__main__":
    ts = arma_wrapper(np.arange(1,40), 40, 10)

    ts.auto_fit((2,2), "aic", False).stats

    print(len(ts.forecast))

    ts.plot_forecast("0")