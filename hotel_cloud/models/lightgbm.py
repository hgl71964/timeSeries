from lightgbm import train
from lightgbm import Booster
from lightgbm import Dataset
from pandas import DataFrame
from typing import List


"""
low level interface to XGboost
model Booster; train via xgb.train
"""

def lgb_train(train_df: DataFrame, 
        test_df: DataFrame, 
        target: str, 
        param: dict,
        cat_list: List[str],  # categorical feature names
        n_estimators: int = 10,  # num_boost_round
        ) -> Booster:
    """
    list of possible params: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    Return:
        Booster
    """
    if not isinstance(train_df, DataFrame):
        raise TypeError("must provide pf")
    
    # make core data structure TODO make categorical label
    feats = [i for i in train_df.columns if i != target]
    dtrain = Dataset(train_df[feats], label=train_df[target])

    if test_df is not None:
        dtest = Dataset(test_df[feats], label=test_df[target])
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    else:
        dtest = None
        watchlist = [(dtrain, 'train')]

    # make params
    if "max_depth" not in param:
        param.update({"max_depth":6})
    if "eta" not in param:
        param.update({"eta":1e-1})

    return train(param, dtrain, n_estimators, watchlist, categorical_feature=cat_list)
