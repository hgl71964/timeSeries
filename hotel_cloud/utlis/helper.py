import numpy as np
import pandas as pd 
import datetime
from typing import List
from glob2 import glob
from sklearn.model_selection import KFold
from collections import OrderedDict
import os 

class cv_helper:

    def __init__(self,
                name: str, # xgb or lgb
                data_dict: dict,  # index -> date
                labels: list,  # outcome of clustering
                group_num: int,  # if -1 means we use all data
                param: OrderedDict,  #  for xgboost or lightgbm; order for tuning
                cat_list: List[str],  # list of categorical data
                n_estimators: int,  # num_boost_round
                nfold: int, 
                training_func: callable,  # xgb_train or lgb_train
                predict_func: callable,  # xgb_predict or lgb_predict
                ts: object,  #  timeSeries_data object
                metric: object,  # metric object 
                preserved_cols: List[str], 
                target: str, 
                history: int, 
                lag_bound: tuple, 
                **train_kwargs, 
                ):
        self.name = name
        self.data_dict = data_dict
        self.labels = labels
        self.group_num = group_num
        self.param = param
        self.cat_list = cat_list
        self.n_estimators = n_estimators
        self.nfold = nfold
        self.training_func = training_func
        self.predict_func = predict_func
        self.ts = ts
        self.metric = metric
        self.preserved_cols = preserved_cols
        self.target = target
        self.history = history
        self.lag_bound = lag_bound
        self.train_kwargs = train_kwargs

    def run_cv(self, df):
        return cv_helper.CV(df, self.name, self.data_dict, \
                    self.labels, self.group_num, self.param,\
                    self.cat_list, self.n_estimators, self.nfold,\
                    self.training_func, self.predict_func, self.ts,\
                    self.metric, self.preserved_cols, self.target, self.history,\
                    self.lag_bound, **self.train_kwargs)

    def update_param(self, new_param: dict):
        try:
            self.param.update(new_param)
        except:
            raise ValueError("cannot overwrite param")
        # else:
        #     print("overwrite params")


    @staticmethod
    def CV(df: pd.DataFrame,  # df contains all staydates that we want
        name: str, # xgb or lgb
        data_dict: dict,  # index -> date
        labels: list,  # outcome of clustering
        group_num: int,  # if -1 means we use all data
        param: OrderedDict,  #  for xgboost or lightgbm
        cat_list: List[str],  # list of categorical data
        n_estimators: int,  # num_boost_round
        nfold: int, 
        training_func: callable,  # xgb_train or lgb_train
        predict_func: callable,  # xgb_predict or lgb_predict
        ts: object,  #  timeSeries_data object
        metric: object,  # metric object 
        preserved_cols: List[str], 
        target: str, 
        history: int, 
        lag_bound: tuple, 
        **train_kwargs, 
        ):
        """
        hand-made cross-validation
        """
        if isinstance(labels, list):
            all_indices = np.where(np.array(labels)==group_num)[0].flatten()
        else:
            all_indices = np.where(labels==group_num)[0].flatten()
        
        group_size = len(all_indices)

        kf = KFold(n_splits=nfold, shuffle=False)
        softdtw_collector, mse_collector = [[None]*8]*nfold, [[None]*8]*nfold  # [["name", "metric", "group_label", "group_size","nfold","min", "max", "mean"], ...]

        for index, (train_keys, test_keys) in enumerate(kf.split(all_indices)):  # CV

            train_indices, test_indices = all_indices[train_keys], all_indices[test_keys]

            train_dates, test_dates = [data_dict[i] for i in train_indices], \
                                        [data_dict[i] for i in test_indices]
            train_df = ts.make_lag_from_dates(df, train_dates, preserved_cols, \
                                            target, history, lag_bound)
            test_df = ts.make_lag_from_dates(df, test_dates, preserved_cols,\
                                            target, history,lag_bound)

            """here test_df is added to watchlist"""
            bst = training_func(train_df, test_df, target, param, \
                            cat_list, n_estimators, **train_kwargs)

            """apply metric"""
            temp_softdtw, temp_mse = [], []

            # TODO find worst date
            for test_date in test_dates:

                ivd_test_df = ts.make_lag_from_dates(df, [test_date], preserved_cols,\
                                            target, history,lag_bound)
                
                preds = predict_func(ivd_test_df, cat_list, target, bst)
                
                soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
                mse_res = metric.mse(preds, ivd_test_df[target])

                temp_softdtw.append(soft_dtw_res)
                temp_mse.append(mse_res)

            softdtw_collector[index], mse_collector[index] = [name, "softdtw", group_num, group_size, nfold, min(temp_softdtw), max(temp_softdtw), sum(temp_softdtw)/len(temp_softdtw)], \
                                                            [name, "mse", group_num, group_size, nfold, min(temp_mse), max(temp_mse), sum(temp_mse)/len(temp_mse)]

        return pd.DataFrame(softdtw_collector, columns=["name", "metric", "group_label", "group_size", "nfold", "min", "max", "mean"], index=[f"cv_{i}" for i in range(len(softdtw_collector))]), \
                pd.DataFrame(mse_collector, columns=["name", "metric", "group_label", "group_size", "nfold", "min", "max", "mean"], index=[f"cv_{i}" for i in range(len(mse_collector))])


class helper:

    
    @staticmethod
    def _average_over_folds(df):
        df["max_of_maxes"] = df.max()["max"]
        df["min_of_mines"] = df.min()["min"]
        df["mean_of_means"] = df.mean()["mean"]
        df = df.drop(columns=["min", "max", "mean"]).iloc[0]
        return df
    
    @staticmethod
    def _add_weighted_mean(df, num_groups, metric_name):

        size_recorder, mean_recorder = [], []        
        temp_df = df[df["metric"]==f"{metric_name}"]
        for i in range(num_groups):

            size_recorder.append(   \
                    float(temp_df[temp_df["group_label"]==i]["group_size"].to_numpy()))
            mean_recorder.append(   \
                    float(temp_df[temp_df["group_label"]==i]["mean_of_means"].to_numpy()))

        size_recorder, mean_recorder = np.array(size_recorder), np.array(mean_recorder)
        fraction = size_recorder/size_recorder.sum()
        df[f"{metric_name}_weighted_mean"] = np.multiply(fraction, mean_recorder).sum()
        return df


    @staticmethod
    def post_process(num_groups: int,  # numbers of group in total 
                    *args):

        """
        make sure feature names are consistent (do not rename existing feature...)
        """

        softdtw_df, mse_df = [], []

        for df in args:

            # add stats over folds, -> pd.series
            series = helper._average_over_folds(df)

            if series["metric"] == "softdtw":
                softdtw_df.append(series)
            elif series["metric"] == "mse":
                mse_df.append(series)

        if len(softdtw_df) == 1:  # on the unclustered data, i.e. group_num = -1
            return pd.concat(softdtw_df+mse_df, axis=1).T
        else:
            df1 = helper._add_weighted_mean(pd.concat(softdtw_df,axis=1).T, num_groups, "softdtw")
            df2 = helper._add_weighted_mean(pd.concat(mse_df,axis=1).T, num_groups, "mse")
            return pd.concat([df1, df2], axis=0)



class logger:

    @staticmethod
    def save_df(index: int, name:str, params: dict, *args):

        if "name" not in params:
            params["name"] = name  # e.g. xgboost
        
        zidx = str(index).zfill(3)  # 3 figs naming

        file_present = glob(f"./data/log/{zidx}_param.csv") or \
                    glob(f"./data/log/{zidx}_metric.csv")
        
        if file_present:
            raise FileExistsError(f"file No. {zidx} exists!")
        else:
            """save params"""
            temp = pd.DataFrame(list(params.items())).T
            header = temp.iloc[0]
            temp = temp[1:]
            temp.columns = header
            pd.DataFrame(temp).to_csv(f"./data/log/{zidx}_param.csv")  

            """save cv logs"""
            if isinstance(args[0], pd.Series):
                pd.concat(args, axis=1).T.reset_index(drop=True).to_csv(f"./data/log/{zidx}_metric.csv")
            elif isinstance(args[0], pd.DataFrame):
                pd.concat(args, axis=0).to_csv(f"./data/log/{zidx}_metric.csv")
            
            print("save complete")
        return None

    @staticmethod
    def show_df(file_path: str):
        return None


    @staticmethod
    def show_all_df(dir_path: str,  # path to the directory  
                        ):

        files = os.listdir(dir_path)

        param_df, metric_df = [], []

        for f in sorted(files, key=lambda x:x[:3]):  # 3 fig naming
            full_path = os.path.join(dir_path, f)

            if "param" in f:
                param_df.append(pd.read_csv(full_path))
            elif "metric" in f:
                metric_df.append(pd.read_csv(full_path))

        return pd.concat(param_df, axis=0), pd.concat(metric_df, axis=0).reset_index(drop=True).drop(columns=["Unnamed: 0"])