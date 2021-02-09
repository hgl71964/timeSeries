import warnings
import datetime
import numpy as np
import pandas as pd 
from glob2 import glob
from typing import List
from collections import deque 
from tslearn.clustering import TimeSeriesKMeans

"""
specific for hotel cloud data
"""

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
                preserved_col: List[str],  # feats in the modelling
                data_range: tuple,  #  e.g. (2018, 2020) -> use staydate in 2018-2020
                target: str,  # the target to model
                history: int,  # length of the booking curve to be used
                filter_all_zero: bool,  # whether to filter ts that is shorter than history
                lag_feats,
                lag_range,
                **kwargs,  # handle interpolation method
                ):
        """
        cleanse the dataFrame and return

        Return: 
            data: np.ndarray; each row is a booking curve for a staydate
            data_dict: dict; index -> staydate
            df
        """

        start_date, end_data = datetime.datetime(data_range[0], 1, 1, 0, 0), \
                                    datetime.datetime(data_range[1]+1, 1, 1, 0, 0)
        num_days = (end_data - start_date).days
        data_dict, idx = {}, 0
        
        # deque with O(1) complexity to append
        all_booking_curve, clean_df = deque(), deque()
        for _ in range(num_days):

            full_date = start_date.strftime("%Y-%m-%d")
            start_date += datetime.timedelta(days=1)

            # s_df -> df of a specific stay_date
            s_df = df[(df["staydate"] == full_date)].groupby("lead_in").sum()\
                                        .filter(preserved_col)

            # filter
            if filter_all_zero and len(s_df[target]) < history:  # all_booking_curve less than history are dicarded
                continue

            # apply interpolation
            s_df = self._interpolate(s_df, **kwargs)

            # add feats
            s_df["staydate"] = full_date

            # make lag && temporal info
            s_df = self.make_lag_for_df(s_df, target, lag_range, lag_feats)


            # s_df = self.make_rolling

            # select history
            s_df = s_df.iloc[:history]

            # collect
            data_dict[idx] = full_date
            idx+=1
            clean_df.append(s_df)
            d = s_df[target].to_numpy()
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
                df[feat] = df[feat].replace(0, np.nan)\
                            .interpolate(method=inter_method, order=inter_order)
        return df

    def make_rolling_for_df(self,
                            df,
                            target,
                            rolling_feats,
                            rolling_window: int):
        

        return 


    def make_lag_for_df(self,
                        df,  # consists of all staydates we want
                        target: str,
                        lag_range: tuple, 
                        lag_feats: List[str],  # feature that needs to make lag
                        ):
        """ add lag feature for every 'staydate' of the df """
        dates = df["staydate"].unique().astype(str)
        return self.make_lag_from_dates(df, dates, lag_feats, \
                    target, lag_range)

    def make_lag_from_dates(self,
                        df,
                        dates: List[str],
                        lag_feats: List[str],
                        target: str,
                        lag_range: List[int], 
                        ):
        """
        make lag feature for a single staydate; 
                ensure 'staydate' is a unit
        """
        df_timing = "staydate"  
        df_list = [None] * len(dates)

        if target not in lag_feats:
            full_feats = lag_feats + [target]
        else:
            lag_feats.remove(target)
            full_feats = lag_feats + [target]

        for i, date in enumerate(dates):

            s_df = df[(df["staydate"] == date)].groupby("lead_in")\
                        .sum().reset_index().drop(columns=["lead_in"])

            s_df = self._add_lag_features(s_df, full_feats , lag_range)
            s_df = s_df.drop(columns=lag_feats)  # drop no lag features
            s_df = self._add_temporal_info(s_df, date)
            s_df = s_df.dropna()  # remove row has NA
            s_df[df_timing] = date  
            df_list[i] = s_df

        return pd.concat(df_list, axis=0, ignore_index=True)

    def _add_lag_features(self, 
                    df, 
                    features, 
                    lag_range: List[int],
                    ):
        for feat in features:
            for lag in lag_range: 
                df[f"{feat}_lag_{lag}"] = df[feat].shift(-lag) 
        return df
 
    def _add_temporal_info(self, 
                           df,
                           date):
        date_time = datetime.datetime.strptime(date, "%Y-%m-%d")
        _, month, day_of_month, _, _, _, day_of_week, _, _ = date_time.timetuple()
        df["month"] = month
        df["day_of_month"] = day_of_month
        df["day_of_week"] = day_of_week
        # df["day_of_year"] = day_of_year  # this val is unique for every data point in 2019
        return df

    def df_from_dates(self,
                    df,
                    dates: List[str],
                    ):
        """ the staydate col is dropped here"""
        df_list = [None] * len(dates)
        for i, date in enumerate(dates):
            df_list[i] = df[df["staydate"]==date].drop(columns=["staydate"])
        return pd.concat(df_list, axis=0, ignore_index=True)

    def adjust_prices(self,
                    df: pd.DataFrame,
                    percentage: float,  # 0.1 -> raise by 10%; -0.1 -> reduce by 10%
                    ):
        # adjust prices 
        df["rateamount"] = df.apply(lambda df: df["rateamount"]*(1 + percentage), axis=1)

        # re-compute median_pc_diff
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeError)  # supress divide by zero            
            df["median_pc_diff"] = df.apply(lambda df: (df["rateamount"] - df["competitor_median_rate"]) \
                             / df["competitor_median_rate"], axis=1)
        return df

    def train_test_dates(self, 
                        labels: np.ndarray,  # outcome of clustering
                        data_dict: dict,
                        test_size: float = 0.2, 
                        group_num: int = -1,                    
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