from xgboost import train
from xgboost import Booster
from xgboost import DMatrix
from pandas import DataFrame
from pandas import get_dummies
from xgboost import cv
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
    
    # one-hot encoding
    train_df, test_df = _one_hot_encoding(train_df, cat_list),\
                        _one_hot_encoding(test_df, cat_list) 

    # make core data structure
    feats = [i for i in train_df.columns if i != target]

    dtrain = DMatrix(train_df[feats], label=train_df[target])
    if test_df is not None:
        dtest = DMatrix(test_df[feats], label=test_df[target])
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    else:
        dtest = None
        watchlist = [(dtrain, 'train')]

    # get train param from kwarg
    early_stopping_rounds = kwargs.get("early_stopping_rounds", None)
    verbose_eval = kwargs.get("verbose_eval", False)

    return train(param, dtrain, n_estimators, watchlist, \
            early_stopping_rounds=early_stopping_rounds, \
            verbose_eval=verbose_eval, \
            )


def _one_hot_encoding(df, names: List[str]):
    """names a list of feature names that need to be one hot encoding"""

    feats = df.columns.to_list()
    for name in names:
        if name in feats:  # if name in df.cloumns then one-hot encoding
            df = df.join(get_dummies(df[name], \
                        prefix=f"{name}")).drop(columns=[name])
    return df