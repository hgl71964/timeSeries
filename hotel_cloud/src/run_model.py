import os 
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from package.utlis.color import bcolors
from package.utlis.folder import folder

from package.component.preprocess import preprocessing

"""GBM"""
from package.utlis.scores import cv_scores
from package.utlis.helper import helper
from package.models.xgboost_ts import xgb_train, xgb_predict
from package.models.lightgbm_ts import lgb_train, lgb_predict
from package.utlis.evaluation import forecast_metric

"""Args"""
cli = argparse.ArgumentParser(description="config global parameters")

cli.add_argument("--lb",
                dest="lb",
                type=int,
                nargs=2, 
                default=(1, 3),
                help="2 args -> lag_bound for lag feats")

cli.add_argument("--nc",
                dest="nc",
                type=int,
                default=5,
                help="number of clusters")

cli.add_argument("--target",
                dest="target",
                type=str,
                default="rooms_all",
                help="forecast target")

cli.add_argument("--gn",
                dest="gn",
                type=int,
                default=-1,
                help="group number to use, -1 means use all available data")

cli.add_argument("--ts",
                dest="ts",
                type=float,
                default=0.2,
                help="testset size")

cli.add_argument("-k",
                dest="k",
                type=int,
                default=5,
                help="number of folds for cross-validation")

cli.add_argument("--history",
                dest="history",
                type=int,
                default=100,
                help="number of history of time series")

args = cli.parse_args()



DIR = folder.get_working_dir("hotel_cloud")      # define working dir folder

YEAR = 2019                     # for check only
STAY_DATE = "01-11"             # for check only

TARGET = args.target            # target for forecasting
HISTORY = args.history          # length of the time series we want to find
DATA_RANGE = (2019, 2019)       # use data from (0 - 1)
TEST_SIZE = args.ts
GROUP_NUM = args.gn
N_CLUSTER = args.nc      # num_clusters are determined by the elbow-point

CAT_LIST = ["month", "day_of_month", "day_of_week"]  # list to categorical data needed to be added
EPOCHS = 256                    # train iterations; early stopping to prevent overfitting
KFOLD = args.k                       # score via 3 fold cross-validation
LAG_RANGE = args.lb              # the bound for lagged features

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


# ------------------------------------------------------------------------------------------
data_files = os.listdir(os.path.join(DIR, "data", "log"))

if "optimal_config.npy" in data_files:
    print(f"{bcolors.HEADER} reading optimal configuration {bcolors.ENDC}")
    opt_config = np.load(os.path.join(DIR, "data", "log", "optimal_config.npy"), \
                        allow_pickle='TRUE').item()

    name = opt_config["name"]
    opt_config.pop("name")

    if name == "xgb":
        xgb_params.update(opt_config)

        # register optimal param
        training_func = xgb_train
        predict_func = xgb_predict
        param = xgb_params
        training_param = xgb_train_params

    elif name == "lgb":
        lgb_param.update(opt_config)

        training_func = lgb_train
        predict_func = lgb_predict
        param = lgb_param
        training_param = lgb_train_param

    print(f"{bcolors.OKGREEN} optimal model && configuration: ")
    print(name)
    print(param)
    print(f"{bcolors.ENDC}")

else:
    print(f"{bcolors.FAIL} cannot load optimal config, using default params \n {bcolors.ENDC}")


df, data_dict, preds, ts = preprocessing(DIR, os.path.join(DIR, "data", "hotel-4_12jan2021.csv"),  \
                YEAR, DATA_RANGE, HISTORY, TARGET, N_CLUSTER, ALL_FEAT, LAG_RANGE)

"""
train test
"""
if GROUP_NUM == -1:  # use all data
    train_dates, test_dates = ts.train_test_dates(np.zeros_like(preds)-1, data_dict, test_size=TEST_SIZE, group_num=GROUP_NUM)
else:
    train_dates, test_dates = ts.train_test_dates(preds, data_dict, test_size=TEST_SIZE, group_num=GROUP_NUM)

print(f"{bcolors.INFO_CYAN}trainset size: {len(train_dates)} \t \
                        testset size: {len(test_dates)} {bcolors.ENDC}")

train_df, test_df = ts.make_lag_from_dates(df, train_dates, ALL_FEAT,\
                        target=TARGET, history=HISTORY, lag_range=LAG_RANGE), \
                        ts.make_lag_from_dates(df, test_dates, ALL_FEAT,\
                        target=TARGET, history=HISTORY, lag_range=LAG_RANGE)

print(cv_scores.CV(df, name, data_dict, np.zeros_like(preds)-1, -1, param, CAT_LIST, EPOCHS, KFOLD, \
          training_func, predict_func, ts, forecast_metric, ALL_FEAT, TARGET, HISTORY, LAG_RANGE, **training_param))

# bst = training_func(train_df, test_df, TARGET, param, CAT_LIST, EPOCHS, **training_param)

# # feature scores
# d = bst.get_score(importance_type="weight")
# print(sorted([(key, val) for key, val in d.items()], key=lambda x:x[-1], reverse=True))

# # cor features
# corr_df = train_df.corr().abs()
# upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))

# for i in range(5):
#     print(upper.iloc[i])


# if False:
#     predict_func(test_df, CAT_LIST, TARGET, bst).shape

