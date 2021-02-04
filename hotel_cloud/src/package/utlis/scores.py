import numpy as np
import pandas as pd 
from typing import List
from sklearn.model_selection import KFold
from copy import deepcopy

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
                target: str, 
                **train_kwargs, 
                ):
        self.name = name
        self.data_dict = data_dict
        self.labels = labels
        self.group_num = group_num
        self.param = deepcopy(param)  # mutable
        self.cat_list = cat_list
        self.n_estimators = n_estimators
        self.nfold = nfold
        self.training_func = training_func
        self.predict_func = predict_func
        self.ts = ts
        self.metric = metric
        self.target = target
        self.train_kwargs = deepcopy(train_kwargs) # mutable

        """register str -- integerd-coded pairs"""
        self.xgb_booster = {"gbtree": 0,
                        "gblinear": 1,
                        "dart": 2, 
                        }
        self.xgb_rev_booster =  {0: "gbtree",
                            1: "gblinear",
                            2: "dart", 
                            }

        self.lgb_booster = {
                            "gbdt":0, 
                            "rf": 1, 
                            "dart": 2, 
                            "goss": 3, 
                            }
        
        self.lgb_rev_booster = {
                            0: "gbdt",
                            1: "rf", 
                            2: "dart", 
                            3: "goss",  
                            }

    def update_param(self, new_param: dict):
        try:
            self.param.update(new_param)
        except:
            raise RuntimeError("cannot overwrite param")
    
    def numeric_to_dict(self, new_vals):
        """
        WARNINGs: order must be correct

        Categorical variables is int-ed before convert to param_dict
        """
        new_param = {}
        if self.name == "xgb":
            new_param["eta"] = new_vals[0]
            new_param["max_depth"] = int(new_vals[1])
            new_param["min_child_weight"] = int(new_vals[2])
            new_param["subsample"] = new_vals[3]
            new_param["lambda"] = new_vals[4]
            # new_param["booster"] = self.xgb_rev_booster[int(new_vals[0])]

        elif self.name == "lgb":
            new_param["eta"] = new_vals[0]
            new_param["num_leaves"] = int(new_vals[1])
            new_param["feature_fraction"] = new_vals[2]
            new_param["subsample"] = new_vals[3]
            new_param["lambda_l2"] = new_vals[4]
            new_param["subsample_freq"] = int(new_vals[5])  
            # new_param["boosting"] = self.lgb_rev_booster[int(new_vals[0])]
        else:
            raise AttributeError(f"{self.name} must be xgb or lgb to generate correct numerical list")
        return new_param

    @property
    def dict_to_numeric(self):
        """WARNINGs: order must be correct"""
        if self.name == "xgb":
            return [
                    self.param["eta"], \
                    self.param["max_depth"], \
                    self.param["min_child_weight"], \
                    self.param["subsample"], \
                    self.param["lambda"], \
                    # self.xgb_booster[self.param["booster"]],  # str -> integer
                    ]
        elif self.name == "lgb":
            return [
                    self.param["eta"],  # float
                    self.param["num_leaves"],  # int
                    self.param["feature_fraction"],  # float 
                    self.param["subsample"],  # float
                    self.param["lambda_l2"], 
                    self.param["subsample_freq"],  # int 
                    # self.lgb_booster[self.param["boosting"]], # int
                    ]
        else:
            raise AttributeError(f"{self.name} must be xgb or lgb to generate correct numerical list")


    def run_cv(self, df):
        return cv_scores.CV(df, self.name, self.data_dict, \
                    self.labels, self.group_num, self.param,\
                    self.cat_list, self.n_estimators, self.nfold,\
                    self.training_func, self.predict_func, self.ts,\
                    self.metric, self.target, **self.train_kwargs)

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
        target: str, 
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
        softdtw_collector, mse_collector = [[None]*9]*nfold, [[None]*9]*nfold  # [["name", "metric", "group_label", "group_size","nfold","min", "max", "mean", "std"], ...]

        for index, (train_keys, test_keys) in enumerate(kf.split(all_indices)):  # CV

            train_indices, test_indices = all_indices[train_keys], all_indices[test_keys]

            train_dates, test_dates = [data_dict[i] for i in train_indices], \
                                        [data_dict[i] for i in test_indices]
            train_df = ts.dataset_from_dates(df, train_dates)
            test_df = ts.dataset_from_dates(df, test_dates)

            """here test_df is added to watchlist"""
            bst = training_func(train_df, test_df, target, param, \
                            cat_list, n_estimators, **train_kwargs)

            """apply metric"""
            temp_softdtw, temp_mse = [], []

            for test_date in test_dates:

                ivd_test_df = ts.dataset_from_dates(df, [test_date])
                
                preds = predict_func(ivd_test_df, cat_list, target, bst)
                
                soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
                mse_res = metric.mse(preds, ivd_test_df[target])

                temp_softdtw.append(soft_dtw_res)
                temp_mse.append(mse_res)

            softdtw_collector[index], mse_collector[index] = [name, "softdtw", group_num, group_size, nfold, min(temp_softdtw), max(temp_softdtw), np.mean(temp_softdtw), np.std(temp_softdtw)], \
                                                            [name, "mse", group_num, group_size, nfold, min(temp_mse), max(temp_mse), np.mean(temp_mse), np.std(temp_mse)]

        return pd.DataFrame(softdtw_collector, columns=["name", "metric", "group_label", "group_size", "nfold", "min", "max", "mean", "std"], index=[f"cv_{i}" for i in range(len(softdtw_collector))]), \
                pd.DataFrame(mse_collector, columns=["name", "metric", "group_label", "group_size", "nfold", "min", "max", "mean", "std"], index=[f"cv_{i}" for i in range(len(mse_collector))])

    
    @staticmethod
    def _CV_average_over_folds(df):
        df["max_of_maxes"] = df.max()["max"]
        df["min_of_mines"] = df.min()["min"]
        df["mean_of_means"] = df.mean()["mean"]
        df = df.drop(columns=["min", "max", "mean"]).iloc[0]
        return df
    
    @staticmethod
    def _CV_add_weighted_mean(df, num_groups, metric_name):

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
    def CV_post_process(num_groups: int,  # numbers of group in total
                    *args):

        """
        make sure feature names are consistent (do not rename existing feature...)
        """
        softdtw_df, mse_df = [], []

        for df in args:

            # add stats over folds, -> pd.series
            series = cv_scores._CV_average_over_folds(df)

            if series["metric"] == "softdtw":
                softdtw_df.append(series)
            elif series["metric"] == "mse":
                mse_df.append(series)

        if len(softdtw_df) == 1:  # on the unclustered data, i.e. group_num = -1
            return pd.concat(softdtw_df+mse_df, axis=1).T
        else:
            df1 = cv_scores._CV_add_weighted_mean(pd.concat(softdtw_df,axis=1).T, num_groups, "softdtw")
            df2 = cv_scores._CV_add_weighted_mean(pd.concat(mse_df,axis=1).T, num_groups, "mse")
            return pd.concat([df1, df2], axis=0)
