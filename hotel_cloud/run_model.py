import os 
import argparse
import logging
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
cli = argparse.ArgumentParser(description="config global parameters")

cli.add_argument("--lb",
                dest="lb",
                type=int,
                nargs=2, 
                default=(1, 3),
                help="2 args -> lag_bound for lag feats")

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
                default=3,
                help="number of folds for cross-validation")

cli.add_argument("--history",
                dest="history",
                type=int,
                default=100,
                help="number of history of time series")

args = cli.parse_args()

HOME = os.path.expanduser("~")  # define home folder
YEAR = 2019                     # for check only
STAY_DATE = "01-11"             # for check only

TARGET = args.target            # target for forecasting
HISTORY = args.history          # length of the time series we want to find
DATA_RANGE = (2019, 2019)       # use data from (0 - 1)
TEST_SIZE = args.ts
GROUP_NUM = args.gn

CAT_LIST = ["month", "day_of_month", "day_of_week"]  # list to categorical data needed to be added
EPOCHS = 256                    # train iterations; early stopping to prevent overfitting
KFOLD = args.k                       # score via 3 fold cross-validation
LAG_FEAT = args.lb              # the bound for lagged features

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
data_files = os.listdir(os.path.join(HOME, "data", "log"))

if "optimal_config.npy" in data_files:
    print(f"{bcolors.HEADER} reading optimal configuration {bcolors.ENDC}")
    opt_config = np.load(os.path.join(HOME, "data", "log", "optimal_config.npy"), \
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


"""
data reading
"""
raw_df = pd.read_csv("~/data/hotel-4_12jan2021.csv") 
raw_df["reportdate"] = raw_df["reportdate"].astype("datetime64[ns]")
raw_df["staydate"] = raw_df["staydate"].astype("datetime64[ns]")
t = raw_df["staydate"].unique().shape[0]
print(f"{bcolors.INFO_CYAN}staydate has {t} days {bcolors.ENDC}")

"""
data cleansing 
"""
ts = timeSeries_data(**{"year": YEAR, })
data, data_dict, df = ts.cleansing(raw_df, DATA_RANGE, TARGET, \
                    HISTORY, True, **{"interpolate_col": [TARGET]})

print(f"{bcolors.INFO_CYAN}target shape", data.shape)


"""
reading clustering file
"""
data_files = os.listdir(os.path.join(HOME, "data", "log"))
if "preds.npy" in data_files:
    print(f"{bcolors.HEADER}reading from data folder... {bcolors.ENDC}")
    preds = np.load(os.path.join(HOME, "data", "log","preds.npy"))


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
                        target=TARGET, history=HISTORY, lag_bound=LAG_FEAT), \
                        ts.make_lag_from_dates(df, test_dates, ALL_FEAT,\
                        target=TARGET, history=HISTORY, lag_bound=LAG_FEAT)


bst = training_func(train_df, test_df, TARGET, param, CAT_LIST, EPOCHS, **training_param)

# feature scores
d = bst.get_score(importance_type="weight")
print(sorted([(key, val) for key, val in d.items()], key=lambda x:x[-1], reverse=True))

# cor features
corr_df = train_df.corr().abs()
upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))

for i in range(5):
    print(upper.iloc[i])




if False:
    predict_func(test_df, CAT_LIST, TARGET, bst).shape
