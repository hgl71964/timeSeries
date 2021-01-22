import numpy as np
import pandas as pd 
from tslearn.clustering import TimeSeriesKMeans
import datetime
from typing import List
from glob2 import glob
from collections import deque 


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

    def cleansing(self, 
                df, 
                years: tuple,  #  e.g. (2018, 2020) -> use staydate in 2018-2020
                target: str,  # the target to model
                history: int = 100,  # length of the booking curve to be used 
                filter_all_zero=True,  # whether to filter ts that is shorter than history 
                **kwargs,  # handle interpolation method
                ):
        """
        cleanse the dataFrame and return 

        Return: 
            data: np.ndarray; each row is a booking curve for a staydate
            data_dict: dict; index -> staydate
            df
        """

        start_date, end_data = datetime.datetime(years[0], 1, 1, 0, 0), datetime.datetime(years[1]+1, 1, 1, 0, 0)
        num_days = (end_data - start_date).days
        data_dict, idx = {}, 0
        
        # deque with O(1) complexity to append
        all_booking_curve, clean_df = deque(), deque()
        for i in range(num_days):

            full_date = start_date.strftime("%Y-%m-%d")
            start_date += datetime.timedelta(days=1)

            s_df = df[(df["staydate"] == full_date)].groupby("lead_in").sum().iloc[:history]

            if filter_all_zero and len(s_df[target]) < history:  # all_booking_curve less than history are dicarded
                continue

            # apply interpolation
            s_df = self._interpolate(s_df, **kwargs)

            # add feats
            s_df["staydate"] = full_date
            d = s_df[target].to_numpy()  

            # collect
            data_dict[idx] = full_date
            idx+=1
            clean_df.append(s_df)
            all_booking_curve.append(d)

        # type conversion & post-cleansing
        all_booking_curve = np.flip(np.array(all_booking_curve).reshape(idx, -1), axis=1)
        all_booking_curve = np.where(all_booking_curve < 0, 0, all_booking_curve)  # if there exists negative term due to interpolation 

        assert (np.all(np.isfinite(all_booking_curve)) and not np.any(np.isnan(all_booking_curve))), \
                                        "data contain NAN or INF"

        return all_booking_curve, data_dict, pd.concat(clean_df, axis=0)

    def _interpolate(self, df, **kwargs):

        feats, interpolate_param = kwargs.get("interpolate_col", []), \
                                kwargs.get("interpolate_param", ("spline", 3))
        if not feats:
            return df

        # TODO interpolate logic can be improved 
        inter_method, inter_order = interpolate_param

        if isinstance(feats, str):  # prevent keyError
            feats = [feats]
        
        for feat in feats:  # interpolate "0" for all feats in the list

            # interpolation algorithm cannot handle last entry
            if df[feat].eq(0).iloc[0]:
                df[feat].iat[0] = df[feat].iloc[1]  # direct assignment

            # only iterpolate if the last 20 dates contain 0 
            if any(df[feat].iloc[:20].eq(0)):  
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
        # df["day_of_year"] = day_of_year  # this val is unique for every data point in 2019
        return df

    def train_test_dates(self, 
                        labels: np.ndarray,  # outcome of clustering 
                        data_dict: dict,
                        test_size: float = 0.2, 
                        group_num: int = 0,                    
                        ):

        all_indices = np.where(labels==group_num)[0].flatten()

        n = len(all_indices)
        testSize = int(n * test_size)

        test_indices = np.random.choice(all_indices, testSize, replace=False)
        train_indices = [i for i in all_indices if i not in test_indices]

        test_dates=[None] * testSize
        for i, key in enumerate(test_indices):
            test_dates[i] = data_dict[int(key)]

        train_dates = [None] * int(n - testSize)
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
