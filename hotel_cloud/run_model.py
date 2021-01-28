import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utlis.color import bcolors

"""data cleansing"""
from utlis.timeSeries_dataset import timeSeries_data

"""GBM"""
from utlis.scores import cv_scores
from models.xgboost_ts import xgb_train, xgb_predict
from models.lightgbm_ts import lgb_train, lgb_predict
from utlis.evaluation import forecast_metric

"""Args"""
HOME = os.path.expanduser("~")  # define home folder 
YEAR = 2019                     # for check only 
STAY_DATE = "01-11"             # for check only 


TARGET = "rooms_all"            # target for forecasting 
HISTORY = 100                   # length of the time series we want to find 
DATA_RANGE = (2019, 2019)       # use data from 2018 - 2019
N_CLUSTER = 5                   # num_clusters are determined by the elbow-point

CAT_LIST = ["month", "day_of_month", "day_of_week"]  # list to categorical data needed to be added
EPOCHS = 256                    # train iterations; early stopping to prevent overfitting 
KFOLD = 3                       # score via 3 fold cross-validation  
LAG_FEAT = (1, 3)               # the bound for lagged features

ALL_FEAT = ["rooms_all", #"is_holiday_staydate", #"revenue_all", "adr_all",  
            "google_trend_1_reportdate", "google_trend_2_reportdate", 
            "google_trend_1_staydate", "google_trend_2_staydate", 
            "competitor_median_rate", "competitor_max_rate", "competitor_min_rate",
            "rateamount_mean", "rateamount",
            "median_pc_diff", #"total_roomcount"
            ]

xgb_params = {
        # General parameters
        "booster": "gbtree",
        "verbosity": 0,  # 0 (silent), 1 (warning), 2 (info), and 3 (debug)
        # Booster parameters
        "eta": 1e-1,  # aka learning rate
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1, 
        "lambda": 1, # l2 regularisation
        # "gamma": 0,
        # Learning task parameters
        "objective": "reg:squarederror",
        "eval_metric": "rmse",  # list, then the last one will be used for early stopping 
          }
xgb_train_params = {
              "verbose_eval": False,
              "early_stopping_rounds": 20, 
              }

lgb_param = {
    "boosting": "gbdt",
    "objective": "rmse",
    "metric": {"rmse"},  # can be a list of metric 
    "first_metric_only" : True,  # use only the first metric for early stopping
    # -----
    "eta": 0.05,  
    "num_leaves": 31,
    "feature_fraction": 0.9, # select a subset of feature to train
    "subsample": 0.8,    # aka bagging
    "subsample_freq": 5,  # aka bagging frequency
    "lambda_l2": 0,
    "verbose": 0, 
          }
lgb_train_param = {
              "verbose_eval": False,
              "early_stopping_rounds": 20, 
              }
"""end of Args"""

data_files = os.listdir(os.path.join(HOME, "data"))

if "optimal_config.npy" in data_files:
    print(f"{bcolors.HEADER} reading optimal configuration {bcolors.ENDC}")
    opt_config = np.load(os.path.join(HOME, "data", "optimal_config.npy"), 
                        allow_pickle='TRUE').item()
    
    print(f"{bcolors.INFO_CYAN} optimal configuration: ")
    print(opt_config)
    print(f"{bcolors.ENDC}")

else:
    print(f"{bcolors.FAIL} cannot load optimal config, using default setting {bcolors.ENDC}")




