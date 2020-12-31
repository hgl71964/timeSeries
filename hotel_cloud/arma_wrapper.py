# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
import pandas as pd


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
                    mod = sm.tsa.statespace.SARIMAX(self.arr, order=(i,0,j), enforce_invertibility=False)
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
        
        self.model = sm.tsa.statespace.SARIMAX(self.arr, order=(p, 0, q))
        self.res = self.model.fit()
        return self
    
    def forecast(self,):
        return self.res.forcast(step=self.forecast_len) if self.res is not None else print("haven't fit model!")

    @property
    def stats(self,):
        print("data len: ", len(self.arr))
        print("forecast len: ", self.forecast_len)

        if self.model is not None:
            print(f"model order {self.model.model_orders}")

        if self.res is not None:
            pass # TODO add stats