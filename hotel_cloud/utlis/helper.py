import numpy as np
import pandas as pd 
import datetime
from typing import List
from glob2 import glob
from sklearn.model_selection import KFold
import os 


class helper:

    @staticmethod
    def CV(df: pd.DataFrame,  # df contains all staydates that we want
        name: str, # xgb or lgb
        data_dict: dict,  # index -> date
        labels: list,  # outcome of clustering
        group_num: int,  # if -1 means we use all data
        param: dict,  #  for xgboost or lightgbm
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
        **kwargs, 
        ):
        """
        hand-made cross-validation
        """
        if isinstance(labels, list):
            all_indices = np.where(np.array(labels)==group_num)[0].flatten()
        else:
            all_indices = np.where(labels==group_num)[0].flatten()

        kf = KFold(n_splits=nfold, shuffle=False)
        softdtw_collector, mse_collector = [[None]*7]*nfold, [[None]*7]*nfold  # [["name", "metric", "group_num", "nfold","min", "max", "mean"], ...]

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
                            cat_list, n_estimators, **kwargs)

            """apply metric"""
            temp_softdtw, temp_mse = [], []

            # TODO find worst date
            for test_date in test_dates:

                # TODO record the worst scenario date for visual
                ivd_test_df = ts.make_lag_from_dates(df, [test_date], preserved_cols,\
                                            target, history,lag_bound)
                
                preds = predict_func(ivd_test_df, cat_list, target, bst)
                
                soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
                mse_res = metric.mse(preds, ivd_test_df[target])

                temp_softdtw.append(soft_dtw_res)
                temp_mse.append(mse_res)

            softdtw_collector[index], mse_collector[index] = [name, "softdtw", group_num, nfold, min(temp_softdtw), max(temp_softdtw), sum(temp_softdtw)/len(temp_softdtw)], \
                                                            [name, "mse", group_num, nfold, min(temp_mse), max(temp_mse), sum(temp_mse)/len(temp_mse)]

        return pd.DataFrame(softdtw_collector, columns=["name", "metric", "group_num", "nfold", "min", "max", "mean"], index=[f"cv_{i}" for i in range(len(softdtw_collector))]), \
                pd.DataFrame(mse_collector, columns=["name", "metric", "group_num", "nfold", "min", "max", "mean"], index=[f"cv_{i}" for i in range(len(mse_collector))])

    @staticmethod
    def post_process(*args):

        temp = []        

        for df in args:

            df["max_of_maxes"] = df.max()["max"]
            df["min_of_mines"] = df.min()["min"]
            df["mean_of_means"] = df.mean()["mean"]
            df = df.drop(columns=["min", "max", "mean"]).iloc[0]

            temp.append(df)

        return temp



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
            temp = pd.DataFrame(list(params.items())).T
            header = temp.iloc[0]
            temp = temp[1:]
            temp.columns = header
            pd.DataFrame(temp).to_csv(f"./data/log/{zidx}_param.csv")  

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