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

cli.add_argument("--nc",
                dest="nc",
                type=int,
                default=5,
                help="number of clusters")

cli.add_argument("--nd",
                dest="nd",
                type=int,
                default=28,
                help="n days ahead forecasting")

cli.add_argument("--target",
                dest="target",
                type=str,
                default="rooms_all",
                help="forecast target")

cli.add_argument("--history",
                dest="history",
                type=int,
                default=60,
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

args = cli.parse_args()

DIR = folder.get_working_dir("hotel_cloud")      # define working dir folder

YEAR = 2019                     # for check only
STAY_DATE = "01-11"             # for check only

NDAYS_AHED = args.nd
TARGET = args.target            # target for forecasting
DATA_RANGE = (2019, 2019)       # use data from (0 - 1)
TEST_SIZE = args.ts
GROUP_NUM = args.gn
N_CLUSTER = args.nc      # num_clusters are determined by the elbow-point

EPOCHS = 256                    # train iterations; early stopping to prevent overfitting
KFOLD = args.k                       # score via 3 fold cross-validation
HISTORY = args.history

ALL_FEAT = ["rooms_all", #"is_holiday_staydate", #"revenue_all", "adr_all",
            "google_trend_1_reportdate", "google_trend_2_reportdate",
            "google_trend_1_staydate", "google_trend_2_staydate",
            "competitor_median_rate", "competitor_max_rate", "competitor_min_rate",
            "rateamount_mean", "rateamount",
            "median_pc_diff", #"total_roomcount"
            "lead_in",
            ]

LAG_FEAT = ["rooms_all", #"is_holiday_staydate", #"revenue_all", "adr_all",
            "google_trend_1_reportdate", "google_trend_2_reportdate",
            "competitor_median_rate", "competitor_max_rate", "competitor_min_rate",
            "rateamount_mean", #"rateamount", -> this maybe set by ourselves
            "median_pc_diff", #"total_roomcount"
            ]
LAG_DAYS = [28, 29, 30, 31, 32]              # the bound for lagged features

ROLLING_WINDOWS = [3, 7, 14]

INTER_FEAT = ["rooms_all", #"is_holiday_staydate", #"revenue_all", "adr_all",
            "google_trend_1_reportdate", "google_trend_2_reportdate",
            "competitor_median_rate", "competitor_max_rate", "competitor_min_rate",
            "rateamount_mean", "rateamount",
            "median_pc_diff", #"total_roomcount"
            ]
INTER_METHODS = ("linear", 1)

# WARNING: change this need to register with XGboost training func
CAT_LIST = ["month", "day_of_month", "day_of_week", "lead_in"]  # list to categorical data

xgb_params = {
        # General parameters
        "booster": "gblinear",
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
    "objective": "mse",  # rmse
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
              "monotone_constraints": True,  # this adds monotonic constraint
              "monotone_constraints_method": "advanced",  # "basic", "intermediate", "advanced"
              }
"""end of Args"""

# ------------------------------------------------------------------------------------------
log_files = os.listdir(os.path.join(DIR, "data", "log"))

if "optimal_config.npy" in log_files:
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
    print(training_param)
    print(f"{bcolors.ENDC}")

else:
    # name = "lgb"
    # training_func = lgb_train
    # predict_func = lgb_predict
    # param = lgb_param
    # training_param = lgb_train_param
    name = "lgb"
    training_func = lgb_train
    predict_func = lgb_predict
    param = lgb_param
    training_param = lgb_train_param
    print(f"{bcolors.FAIL} cannot load optimal config, using default {name} \n {bcolors.ENDC}")

# finish reading
# ------------------------------------------------------------------------------------------


""" pre-processing """
print(f"{bcolors.INFO_CYAN} forecasting {NDAYS_AHED} days ahead {bcolors.ENDC}")

df, data_dict, preds, ts = preprocessing(DIR, os.path.join(DIR, "data", "hotel-4_12jan2021.csv"),  \
                YEAR, DATA_RANGE, HISTORY, NDAYS_AHED, TARGET, N_CLUSTER, \
                ALL_FEAT, LAG_FEAT, LAG_DAYS, ROLLING_WINDOWS,\
                INTER_FEAT, INTER_METHODS)

""" train && test"""
if GROUP_NUM == -1:  # use all data
    print(f"{bcolors.INFO_CYAN} using all data for training {bcolors.ENDC}")
    train_dates, test_dates = ts.train_test_dates(np.zeros_like(list(data_dict.keys()))-1, data_dict, test_size=TEST_SIZE, group_num=GROUP_NUM)
# else:
#     print(f"{bcolors.INFO_CYAN} using grouped data for training {bcolors.ENDC}")
#     train_dates, test_dates = ts.train_test_dates(preds, data_dict, test_size=TEST_SIZE, group_num=GROUP_NUM)

print(f"{bcolors.INFO_CYAN}trainset size: {len(train_dates)} \t \
                        testset size: {len(test_dates)} {bcolors.ENDC}")

train_df, test_df = ts.df_from_dates(df, train_dates), ts.df_from_dates(df, test_dates) 

""" performance evaluation """
print(cv_scores.CV(df, name, data_dict, np.zeros_like(list(data_dict.keys()))-1 , -1, param, CAT_LIST, EPOCHS, KFOLD, \
          training_func, predict_func, ts, forecast_metric, TARGET, **training_param))

# bst = training_func(train_df, test_df, TARGET, param, CAT_LIST, EPOCHS, **training_param)


# if True:
#     pred, real = helper.worst_day_res(test_dates, df, ts, predict_func, CAT_LIST, \
#                                 TARGET, bst, "softdtw", forecast_metric)
#     print(pred)
#     print(real)
#     print(forecast_metric.mse(pred, real))
#     print(helper.feature_important(bst, name, CAT_LIST))


if False:

    origin = helper.price_sensity(test_dates, df, ts, predict_func, CAT_LIST, \
                                    TARGET, bst, 0)

    lower = helper.price_sensity(test_dates, df, ts, predict_func, CAT_LIST, \
                                    TARGET, bst, -0.1)

    higher = helper.price_sensity(test_dates, df, ts, predict_func, CAT_LIST, \
                                    TARGET, bst, 0.1)

    np.save(os.path.join(DIR, "data", "log", "origin.npy"), origin)
    np.save(os.path.join(DIR, "data", "log", "10up.npy"), higher)
    np.save(os.path.join(DIR, "data", "log", "10down.npy"), lower)

if False:
    data = helper.generate_plots(test_dates, df, ts, predict_func, CAT_LIST, \
                                    TARGET, bst, forecast_metric)
    np.save(os.path.join(DIR, "data", "log", "plot_data.npy"), data)



# # cor features
# corr_df = train_df.corr().abs()
# upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))

# for i in range(5):
#     print(upper.iloc[i])

if False:

    predict_func(test_df, CAT_LIST, TARGET, bst).shape

