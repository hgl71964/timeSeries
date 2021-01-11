import numpy as np
import pandas as pd 
from tslearn.clustering import TimeSeriesKMeans


class timeSeries_data:

    def __init__(self, **kwargs):
        self.year = kwargs.get("year", 2019)

    
    def check_staydate(self, 
                        df: pd.DataFrame, 
                        stay_date: str, 
                        preserved_col: list,  # list of feature name to preserve
                        year = None,
                        ):
        if year is None:
            return df[(df["staydate"] == str(self.year) +"-"+ stay_date)].groupby("lead_in").sum().filter(preserved_col)

        if isinstance(year, int):
            year = str(year)
        return df[(df["staydate"] == year +"-"+ stay_date)].groupby("lead_in").sum().filter(preserved_col)

    def check_timeseries(self, history: int = 100):

        start_date = datetime.datetime(self.year, 1, 1, 0, 0)
        dates = [None]*365; dates[0] = start_date.strftime("%m-%d")
        data=np.empty((365, history), dtype=np.float32)

        for i in range(1, 365):
            start_date += datetime.timedelta(days=1)
            dates[i] = start_date.strftime("%m-%d")

        for i in range(365):
            s_df = df[(df["staydate"] == str(self.year) +"-"+ dates[i])].groupby("lead_in").sum()
            d = s_df["rooms_all"].to_numpy()

            if len(d) >=history:
                data[i,:] = np.flip(d[:history])
            else:
                data[i,:] = np.zeros((history, ))
        
        return data
