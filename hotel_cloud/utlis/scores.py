import numpy as np
import pandas as pd 
from typing import List
from sklearn.model_selection import KFold

class cv_scores:

    def __init__(self,
                name: str, # Must Be "xgb" or "lgb"
                data_dict: dict,  # index -> date
                labels: list,  # outcome of clustering
                group_num: int,  # if -1 means we use all data
                param: dict,  #  for xgboost or lightgbm; order for tuning
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

        # str -> integer-encoded
        self.booster = {"gbtree": 0,
                        "gblinear": 1,
                        "dart":2 , 
                        }


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

    @property
    def dict_to_numeric(self):

        """WARNINGs: order must be correct"""

        if self.name == "xgb":
            return [
                    self.booster[self.param["booster"]],  # str -> integer
                    self.param["eta"], \
                    self.param["max_depth"], \
                    self.param["min_child_weight"], \
                    self.param["subsample"], \
                    self.param["lambda"], \
                    ]
        elif self.name == "lgb":
            return []
        else:
            raise AttributeError(f"{self.name} must be xgb or lgb to generate correct numerical list")


    @staticmethod
    def CV(df: pd.DataFrame,  # df contains all staydates that we want
        name: str, # Must Be "xgb" or "lgb"
        data_dict: dict,  # index -> date
        labels: list,  # outcome of clustering
        group_num: int,  # if -1 means we use all data
        param: dict ,  #  for xgboost or lightgbm
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
