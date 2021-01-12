import numpy as np
import pandas as pd 
from tslearn.clustering import TimeSeriesKMeans
import datetime

class timeSeries_data:

    def __init__(self, **kwargs):
        self.year = kwargs.get("year", 2019)
    
    def check_staydate(self,
                        df: pd.DataFrame, 
                        stay_date: str, 
                        preserved_col: list,  # list of feature name to preserve
                        ):
        return df[(df["staydate"] == str(self.year) +"-"+ stay_date)].groupby("lead_in").sum().filter(preserved_col)

    def check_timeseries(self, df, history: int = 100):
        start_date = datetime.datetime(self.year, 1, 1, 0, 0)
        dates = [None]*365; dates[0] = start_date.strftime("%m-%d")
        data = np.empty((365, history), dtype=np.float32)

        for i in range(1, 365):
            start_date += datetime.timedelta(days=1)
            dates[i] = start_date.strftime("%m-%d")

        for i in range(365):
            s_df = df[(df["staydate"] == str(self.year) +"-"+ dates[i])].groupby("lead_in").sum()
            d = s_df["rooms_all"].to_numpy()

            if len(d) >= history:
                data[i,:] = np.flip(d[:history])
            else:
                data[i,:] = np.zeros((history, ))

        return data

    def cleansing(self, df, history: int = 100, filter_all_zero=True, **kwargs):
        """
        this method handles missing values 

        Return: 
            data: np.ndarray; each row is a booking curve for a staydate
            data_dict: dict; index -> staydate
        """
        interpolate_feat, interpolate_param = kwargs.get("interpolate_feat", []), \
                                        kwargs.get("interpolate_param", ("spline", 3))

        start_date, end_data = datetime.datetime(self.year, 1, 1, 0, 0), datetime.datetime(self.year+1, 1, 1, 0, 0)
        num_days = (end_data - start_date).days

        dates = [None]*num_days; dates[0] = start_date.strftime("%m-%d")
        data = np.empty((num_days, history), dtype=np.float32)

        for i in range(1, num_days):
            start_date += datetime.timedelta(days=1)
            dates[i] = start_date.strftime("%m-%d")

        data_dict = {}
        for i in range(num_days):

            full_date = str(self.year) +"-" + dates[i] 
            data_dict[i] = full_date

            s_df = self._interpolate(df[(df["staydate"] == full_date)].groupby("lead_in").sum().iloc[:history], \
                                                        interpolate_feat, interpolate_param)

            d = s_df["rooms_all"].to_numpy()

            if len(d) >= history:
                data[i,:] = np.flip(d[:history])
            else:
                data[i,:] = np.zeros((history, ))

        if filter_all_zero:  # if there are staydates that have all 0 booking curve
            index = []
            for i in range(data.shape[0]):
                if np.all(data[i]==0):
                    index.append(i)
            index = np.array(index)

            for key in index:
                data_dict.pop(key)
            data = data[[i for i in range(365) if i not in index]]

        return data, data_dict

    def _interpolate(self, df, feats, inter_params):
        if not feats:
            return df

        inter_method, inter_order = inter_params
        for i, feat in enumerate(feats):  # interpolate 0 for all feats in the list
            df[feat] = df[feat].replace(0, np.nan).interpolate(method=inter_method, order=inter_order)
        
        return df


    def segmentation(self, data, labels, data_dict):
        """
        make data based on their labels

        Return: 
            dataset: label -> (date, dataset)
        """
        dataset = {}

        for i, pred in enumerate(labels):  # assign label to each staydate


        for i in range(n):
            d

        


        return None


