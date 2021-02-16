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
        **kwargs,
        ) -> Booster:
    """
    list of possible params: https://lightgbm.readthedocs.io/en/latest/Parameters.html
    Return:
        Booster
    """
    if not isinstance(train_df, DataFrame):
        raise TypeError("must provide pf")
    
    feats = [i for i in train_df.columns if i != target]

    dtrain = Dataset(train_df[feats], label=train_df[target], \
                    feature_name=feats, categorical_feature=cat_list)

    if test_df is not None:
        dtest = Dataset(test_df[feats], label=test_df[target], reference=dtrain, \
                        feature_name=feats, categorical_feature=cat_list)
        watchlist = [dtest, dtrain]
    else:
        watchlist = [dtrain]

    verbose_eval = kwargs.get("verbose_eval", False)
    early_stopping_rounds = kwargs.get("early_stopping_rounds", None)

    # optionally add monotonic constraint, controlled by arg in kwarg
    if kwargs.get("monotone_constraints", False):
        mc = [0] * len(list(train_df[feats]))
        for i, item in enumerate(list(train_df[feats])):
            if item == "rateamount":
                mc[i] = -1
        param.update({"monotone_constraints": mc})
        param.update({"monotone_constraints_method": \
                    kwargs.get("monotone_constraints_method", "basic")})

    return train(param, dtrain, n_estimators, valid_sets=watchlist, \
                    early_stopping_rounds=early_stopping_rounds, \
                    verbose_eval=verbose_eval, \
                    categorical_feature=cat_list, \
                    )

def lgb_predict(df,
                cat_list,  # to keep signature the same
                target, 
                booster,
                ):
    feats = [i for i in df.columns if i != target]
    return booster.predict(df[feats])
