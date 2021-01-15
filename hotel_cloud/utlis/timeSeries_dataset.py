import numpy as np
import pandas as pd 
from tslearn.clustering import TimeSeriesKMeans
import datetime
from typing import List

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

    def cleansing(self, df,  history: int = 100, filter_all_zero=True, **kwargs):
        """
        this method handles missing values 

        Return: 
            data: np.ndarray; each row is a booking curve for a staydate
            data_dict: dict; index -> staydate
        """
        interpolate_col, interpolate_param = kwargs.get("interpolate_col", []), \
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
                                        interpolate_col, interpolate_param)

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

        data = np.where(data < 0, 0, data)  # if there exists negative term due to interpolation 
        return data, data_dict

    def _interpolate(self, df, feats, inter_params):
        if not feats:
            return df

        inter_method, inter_order = inter_params
        for feat in feats:  # interpolate 0 for all feats in the list
            if any(df[feat].iloc[:20].eq(0)):  # only iterpolate if the last 20 dates contain 0 
                df[feat] = df[feat].replace(0, np.nan).interpolate(method=inter_method, order=inter_order)
        return df


    def _add_lag_features(self, 
                    df, 
                    features, 
                    lag_bound,
                    ):
        for feat in features:
            for lag in range(lag_bound[0], lag_bound[1]+1):
                df[f"{feat}_lag_{lag}"] = df[feat].shift(-lag) 
        return df
    
    def _add_temporal_info(self, 
                           df,
                           date):
        date_time = datetime.datetime.strptime(date, "%Y-%m-%d")
        _, month, day_of_month, _, _, _, day_of_week, day_of_year, _ = date_time.timetuple()
        df["month"] = month
        df["day_of_month"] = day_of_month
        df["day_of_week"] = day_of_week
        df["day_of_year"] = day_of_year
        return df

    def train_test_dates(self, 
                        labels: np.ndarray,  # outcome of clustering 
                        data_dict: dict,
                        test_size: float = 0.2, 
                        group_num: int = 0,                    
                        ):

        all_indices = np.where(labels==group_num)[0].flatten()

        n = len(all_indices)
        test_size = int(n * 0.2)

        test_indices = np.random.choice(all_indices, test_size, replace=False)
        train_indices = [i for i in all_indices if i not in test_indices]

        test_dates=[None] * test_size
        for i, key in enumerate(test_indices):
            test_dates[i] = data_dict[int(key)]

        train_dates = [None] * int(n - test_size)
        for i, key in enumerate(train_indices):
            train_dates[i] = data_dict[int(key)] 
        return train_dates, test_dates


    def make_lag_from_dates(self, 
                        df, 
                        dates: List[str], 
                        preserved_col: List[str], 
                        target: str,         
                        history: int = 100, 
                        lag_bound: tuple = (2, 4),  # this means we forecast 2 days ahead
                        ):

        """make lag feature for a single staydate"""
        features = [i for i in preserved_col if i != target]  # list of features
        df_list = [None] * len(dates)

        for i, date in enumerate(dates):

            s_df = df[(df["staydate"] == date)].groupby("lead_in")\
                        .sum().reset_index().drop(columns=["lead_in"])\
                                            .filter(preserved_col)
            s_df = self._add_lag_features(s_df, features + [target] , lag_bound)
            s_df = s_df.iloc[:history]
            s_df = s_df.drop(columns = features)  # drop no lag features
            s_df = self._add_temporal_info(s_df, date)
            s_df = s_df.dropna()  # remove row has NA
            df_list[i] = s_df

        return pd.concat(df_list, axis=0, ignore_index=True)