from xgboost import train
from xgboost import Booster
from pandas import DataFrame

"""
low level interface to XGboost
model Booster; train via xgb.train
"""

def xgb_train(train_df: DataFrame, 
        test_df: DataFrame = None, 
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

    if test_df is not None:
        dtest = xgb.DMatrix(test_df, test_df[target])
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    else:
        watchlist = [(dtrain, 'train')]

    # make params
    if "max_depth" not in param:
        param.update({"max_depth":6})
    if "eta" not in param:
        param.update({"eta":1e-1})
    if "silent" not in param:
        param.update({"silent":1})

    return xgb.train(param, dtrain, n_estimators, watchlist)

