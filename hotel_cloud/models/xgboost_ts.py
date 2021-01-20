from xgboost import train
from xgboost import Booster
from xgboost import DMatrix
from pandas import DataFrame
from pandas import get_dummies
from xgboost import cv
from sklearn.model_selection import KFold
from typing import List
import numpy as np



"""
low level interface to XGboost
model Booster; train via xgb.train
"""

def xgb_train(train_df: DataFrame, 
        test_df: DataFrame, 
        target: str, 
        param: dict,
        cat_list: List[str], 
        n_estimators: int = 10,  # num_boost_round
        **kwargs, 
        ) -> Booster:
    """
    list of possible params: https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear
    Return:
        Booster
    """
    if not isinstance(train_df, DataFrame):
        raise TypeError("must provide pf")
    
    # make core data structure
    feats = [i for i in train_df.columns if i != target]
    dtrain = DMatrix(train_df[feats], label=train_df[target])

    if test_df is not None:
        dtest = DMatrix(test_df[feats], label=test_df[target])
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    else:
        dtest = None
        watchlist = [(dtrain, 'train')]

    # TODO add one hot encoding 
    
    # get param from kwarg
    early_stopping_rounds = kwargs.get("early_stopping_rounds", None)
    verbose_eval = kwargs.get("verbose_eval", False)

    return train(param, dtrain, n_estimators, watchlist, \
            early_stopping_rounds=early_stopping_rounds, \
            verbose_eval=verbose_eval, \
            )


def xgb_cv(df: DataFrame,  # df contains all staydates that we want
        data_dict: dict,  # index -> date
        labels: list,
        group_num: int, 
        param: dict,
        cat_list: List[str],  # list of categorical data
        n_estimators: int,  # num_boost_round
        nfold: int, 
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
        bst = xgb_train(train_df, test_df, target, param, \
                        cat_list, n_estimators, **kwargs)

        """apply metric"""
        feats = [i for i in train_df.columns if i != target]

        temp_softdtw, temp_mse = [], []
        for test_date in test_dates:
            ivd_test_df = ts.make_lag_from_dates(df, [test_date], preserved_cols,\
                                         target, history,lag_bound)
            
            preds = bst.predict(DMatrix(ivd_test_df[feats]))
            
            soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
            mse_res = metric.mse(preds, ivd_test_df[target])

            temp_softdtw.append(soft_dtw_res)
            temp_mse.append(mse_res)

        softdtw_collector[index], mse_collector[index] = [min(temp_softdtw), max(temp_softdtw), sum(temp_softdtw)/len(temp_softdtw), "softdtw"], \
                                                        [min(temp_mse), max(temp_mse), sum(temp_mse)/len(temp_mse), "mse"]

    return DataFrame(softdtw_collector, columns=["min", "max", "mean", "metric"], index=[f"cv_{i}" for i in range(len(softdtw_collector))]), \
            DataFrame(mse_collector, columns=["min", "max", "mean", "metric"], index=[f"cv_{i}" for i in range(len(mse_collector))])



def _one_hot_encoding(df, names: List[str]):
    """names a list of feature names that need to be one hot encoding"""

    for name in names:
        df = df.join(get_dummies(df[name], \
                        prefix=f"{name}")).drop(columns=[name])
    return df