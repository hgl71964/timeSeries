import numpy as np
import pandas as pd 
import datetime
from typing import List
from glob2 import glob
from sklearn.model_selection import KFold


class helper:

    @staticmethod
    def CV(df: pd.DataFrame,  # df contains all staydates that we want
        data_dict: dict,  # index -> date
        labels: list,  # outcome of clustering
        group_num: int,  # the group of data that we want to use
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
        softdtw_collector, mse_collector = [[None]*4]*nfold, [[None]*4]*nfold  # [[min, max, mean, name], ...]

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
            for test_date in test_dates:

                # TODO record the worst scenario date for visual
                ivd_test_df = ts.make_lag_from_dates(df, [test_date], preserved_cols,\
                                            target, history,lag_bound)
                
                preds = predict_func(ivd_test_df, cat_list, target, bst)
                
                soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
                mse_res = metric.mse(preds, ivd_test_df[target])

                temp_softdtw.append(soft_dtw_res)
                temp_mse.append(mse_res)

            softdtw_collector[index], mse_collector[index] = [min(temp_softdtw), max(temp_softdtw), sum(temp_softdtw)/len(temp_softdtw), "softdtw"], \
                                                            [min(temp_mse), max(temp_mse), sum(temp_mse)/len(temp_mse), "mse"]

        return pd.DataFrame(softdtw_collector, columns=["min", "max", "mean", "metric"], index=[f"cv_{i}" for i in range(len(softdtw_collector))]), \
                pd.DataFrame(mse_collector, columns=["min", "max", "mean", "metric"], index=[f"cv_{i}" for i in range(len(mse_collector))])


class logger:

    @staticmethod
    def save_df(index: int, name:str, params: dict, *args):

        if "name" not in params:
            params["name"] = name  # e.g. xgboost

        file_present = glob(f"./data/log/param_{index}.csv") or \
                    glob(f"./data/log/metric_{index}.csv")
        
        if file_present:
            raise FileExistsError(f"file No. {index} exists!")
        else:
            temp = pd.DataFrame(list(params.items())).T
            header = temp.iloc[0]
            temp = temp[1:]
            temp.columns = header
            pd.DataFrame(temp).to_csv(f"./data/log/param_{index}.csv")  
            pd.concat(args, axis=0).to_csv(f"./data/log/metric_{index}.csv")
            print("save complete")
        return None

    @staticmethod
    def show_df(file_path: str):
        return None


    @staticmethod
    def show_all_df(path: str,  # path to the directory  
                        ):

        # TODO open all file (df) and concat the results    

        logger.show_df()

        return None