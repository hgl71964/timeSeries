import numpy as np
import torch as tr
from pandas import DataFrame
from src.api_helper import api_utils
dtype = tr.float32


def bayes_loop(
            bayes_opt: object,
            loss_func: callable,  # decorated as api
            df: DataFrame,  # df contains all staydates that we want
            name: str, # xgb or lgb
            data_dict: dict,  # index -> date
            labels: list,  # outcome of clustering
            group_num: int,  # if -1 means we use all data
            param: dict,  #  for xgboost or lightgbm
            cat_list: List[str],  # list of categorical data
            n_estimators: int,  # num_boost_round
            nfold: int, 
            training_func: callable,  # xgb_train or lgb_train
            predict_func: callable,  # xgb_predict or lgb_predict
            ts: object,  #  timeSeries_data object
            metric: object,  # metric object 
            preserved_cols: List[str], 
            target: str, 
            history: int, 
            lag_bound: tuple, 
            **kwargs,  # training params
            ):

    # get x0, y0
    x0 = api_utils.init_query()
    y0 = api_utils.init_reward()

    #  format the initial pair
    x0, y0 = tr.from_numpy(x0).to(device), y0.to(device)

    #  decorate the api
    api = api_utils.api_wrapper(loss_func, metric)

    return bayes_opt.outer_loop(x0, y0, r0, api)

