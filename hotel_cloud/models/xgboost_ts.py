from xgboost import train
from xgboost import Booster
from xgboost import DMatrix
from pandas import DataFrame
from xgboost import cv
from sklearn.model_selection import KFold
from typing import List


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


def xgb_cv(full_df: DataFrame, 
        data_dict: dict,  # index -> date
        param: dict,
        n_estimators: int,  # num_boost_round
        nfold: int, 
        ts: object,  #  a timeSeries_data object
        metric: object,  # for evaluation
        preserved_cols: List[str], 
        target: str, 
        history: int, 
        lag_bound: tuple, 
        **kwargs,
        ):
    """
    hand-made cross-validation
    """
    kf = KFold(n_splits=nfold, shuffle=False)

    softdtw_collector, mse_collector = [[None]*3]*nfold, [[None]*3]*nfold  # min, max, mean

    for index, (train_keys, test_keys) in enumerate(kf.split(list(data_dict.keys()))):  # CV

        train_dates, test_dates = [data_dict[j] for j in train_keys], \
                                    [data_dict[i] for i in test_keys]
        train_df = ts.make_lag_from_dates(full_df, train_dates, preserved_cols, \
                                        target, history, lag_bound, **kwargs)
        test_df = ts.make_lag_from_dates(full_df, test_dates, preserved_cols,\
                                         target, history,lag_bound, **kwargs)

        """here test_df is added to watchlist"""
        bst = xgb_train(train_df, test_df, target, param, n_estimators)

        """apply metric"""
        feats = [i for i in train_df.columns if i != target]

        temp_softdtw, temp_mse = [], []
        for test_date in test_dates:
            ivd_test_df = ts.make_lag_from_dates(full_df, [test_date], preserved_cols,\
                                         target, history,lag_bound, **kwargs)
            
            preds = bst.predict(DMatrix(ivd_test_df[feats]))
            
            soft_dtw_res = metric.softdtw(preds, ivd_test_df[target])
            mse_res = metric.mse(preds, ivd_test_df[target])

            temp_softdtw.append(soft_dtw_res)
            temp_mse.append(mse_res)

        softdtw_collector[index], mse_collector[index] = [min(temp_softdtw), max(temp_softdtw), \
                                                        sum(temp_softdtw)/len(temp_softdtw)], \
                                                        [min(temp_mse), max(temp_mse), \
                                                        sum(temp_mse)/len(temp_mse)]
    # [min, max, mean]
    return softdtw_collector, mse_collector



# def xgb_CV(full_df: DataFrame, 
#         target: str, 
#         param: dict,
#         n_estimators: int = 10,  # num_boost_round
#         nfold=3, 
#         metrics=(),
#         obj=None,
#         feval=None, 
#         early_stopping_rounds=None,
#         verbose_eval=None, 
#         callbacks=None,
#         ) -> Booster:
# 
#     """
#     this does not preserve sequential property
#     """
# 
#     if not isinstance(full_df, DataFrame):
#         raise TypeError("must provide pf")
#     
#     # make core data structure
#     feats = [i for i in full_df.columns if i != target]
#     dcv = DMatrix(full_df[feats], label=full_df[target])
# 
#     # metric = eval_metric 'User can add multiple evaluation metrics' 
#     return cv(param, dcv, num_boost_round=n_estimators, nfold=nfold,
#                 metrics=metrics, obj=obj, feval=feval, 
#                 early_stopping_rounds=early_stopping_rounds,
#                 verbose_eval=verbose_eval, callbacks=callbacks)
    