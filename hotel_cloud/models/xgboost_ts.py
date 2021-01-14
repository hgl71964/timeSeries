import xgboost as xgb
from pandas import DataFrame

"""
dtrain = xgb.DMatrix(d, d["rooms_all"])
dtest = xgb.DMatrix(test_df, test_df["rooms_all"])

# specify parameters via map, definition are same as c++ version
param = {'max_depth':6, 'eta':1, 'silent':1, 'objective':'reg:squarederror'}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
"""

def train(train_df, 
        test_df, 
        target: str, 
        param):

    if not isinstance(train_df, DataFrame):
        raise TypeError("must provide pf")

    dtrain = xgb.DMatrix(train_df, train_df[target])
    dtest = xgb.DMatrix(test_df, test_df[target])


    return None
