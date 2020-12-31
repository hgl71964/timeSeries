# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
import pandas as pd


class arma_wrapper:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forcast_len: int, # length that need to forecast
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

        if forcast_len > n:  # check forcast_len
            raise ValueError("history too short")
        
        
    def auto_fit(self, 
                order_range: tuple,  # (p, q) to search for
                metric: str = "aic",  # metric to score model; "aic" or "bic"
                ):

        p, q = order_range[0], order_range[1]
        scores = np.zeros((p, q))
        self.model = None

        for i in range(p):
            for j in range(q):

                if i==0 and j==0:
                    scores[i][j]=np.inf
                    continue 

                try:
                    mod = sm.tsa.statespace.SARIMAX(self.arr, order=(i,0,j), enforce_invertibility=False)
                    if metric == "aic":
                        scores[i][j] = mod.aic
                    elif metric == "bic":
                        scores[i][j] = mod.bic 
                except:
                    scores[i][j] = np.inf

        p, q = np.where(scores==scores.min()); p, q = int(p), int(q); print(f"optimal (p, q) is {p, q}")
        
        self.model = sm.tsa.statespace.SARIMAX(self.arr, order=(p, 0, q))

        return self
    
    def forecast(self,):

        return 

    def score(self, y_true):

        return None 

