from xgboost import train
from xgboost import Booster
from pandas import DataFrame

"""
low level interface to XGboost
model Booster; train via xgb.train
"""

def xgb_train(train_df: DataFrame, 
        test_df: DataFrame, 
        target: str, 
        param: dict,
        n_estimators: int = 10,  # num_boost_round
        ):

    """
    Return:
        Booster
    """
    if not isinstance(train_df, DataFrame):
        raise TypeError("must provide pf")

    # make core data structure
    dtrain = xgb.DMatrix(train_df, train_df[target])
    dtest = xgb.DMatrix(test_df, test_df[target])

    if "max_depth" not in param:
        param.update({"max_depth":6})
    if "eta" not in param:
        param.update({"eta":6})

    watchlist = [(dtest, 'eval'), (dtrain, 'train')]

    return xgb.train(param, dtrain, n_estimators, watchlist)

