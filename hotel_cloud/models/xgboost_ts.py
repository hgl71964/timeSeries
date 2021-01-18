from xgboost import train
from xgboost import Booster
from xgboost import DMatrix
from pandas import DataFrame
from xgboost import cv


"""
low level interface to XGboost
model Booster; train via xgb.train
"""

def xgb_train(train_df: DataFrame, 
        test_df: DataFrame, 
        target: str, 
        param: dict,
        n_estimators: int = 10,  # num_boost_round
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

    return train(param, dtrain, n_estimators, watchlist)


def CV(full_df: DataFrame, 
        data_dict: dict,  # index -> date
        target: str, 
        param: dict,
        n_estimators: int = 10,  # num_boost_round
        nfold=3, 
        make_folds: callable = lambda x: None,  # return list[list[str]]
        make_lag_from_dates: callable = lambda x: None,  # make lag feature given list of dates
        ):
    
    folds = make_folds(nfold)  # List[List[str]]


    

    return None



def xgb_CV(full_df: DataFrame, 
        target: str, 
        param: dict,
        n_estimators: int = 10,  # num_boost_round
        nfold=3, 
        metrics=(),
        obj=None,
        feval=None, 
        early_stopping_rounds=None,
        verbose_eval=None, 
        callbacks=None,
        ) -> Booster:

    """
    this does not preserve sequential property
    """

    if not isinstance(full_df, DataFrame):
        raise TypeError("must provide pf")
    
    # make core data structure
    feats = [i for i in full_df.columns if i != target]
    dcv = DMatrix(full_df[feats], label=full_df[target])

    # metric = eval_metric 'User can add multiple evaluation metrics' 
    return cv(param, dcv, num_boost_round=n_estimators, nfold=nfold,
                metrics=metrics, obj=obj, feval=feval, 
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval, callbacks=callbacks)
    