"""
this script runs bayes_opt to find optimal hyperparameter for (xg, light)gbm models

the results are store -> ${work_dir}/data/optimal_config.npy
logs are store -> ${work_dir}/data/log/bo*
"""
import os 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from package.utlis.color import bcolors
from package.utlis.folder import folder
from package.component.preprocess import preprocessing

"""GBM"""
from package.utlis.scores import cv_scores
from package.utlis.evaluation import forecast_metric
from package.models.xgboost_ts import xgb_train, xgb_predict
from package.models.lightgbm_ts import lgb_train, lgb_predict

"""BO"""
from package.baye_opt_gbm.bayes_opt import bayes_loop, BO_post_process, bayesian_optimiser

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

cli.add_argument("-t",
                dest="t",
                type=int,
                default=3,
                help="number of iteration of bayes_opt")

cli.add_argument("-k",
                dest="k",
                type=int,
                default=5,
                help="number of folds for cross-validation")

cli.add_argument("--history",
                dest="history",
                type=int,
                default=60,
                help="number of history of time series")

cli.add_argument("--bm",
                dest="bm",
                type=str,
                default="mse",
                help="metric to use in bayes_opt")

args = cli.parse_args()

DIR = folder.get_working_dir("hotel_cloud")      # define working dir folder

YEAR = 2019                     # for check only
STAY_DATE = "01-11"             # for check only

NDAYS_AHED = args.nd
TARGET = args.target        # target for forecasting
HISTORY = args.history                   # length of the time series we want to find
DATA_RANGE = (2019, 2019)       # use data from 2018 - 2019
N_CLUSTER = args.nc      # num_clusters are determined by the elbow-point

EPOCHS = 256                    # train iterations; early stopping to prevent overfitting
KFOLD = args.k                       # score via 3 fold cross-validation

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
            "rateamount_mean", "rateamount",
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


# all params https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear
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

# all params https://lightgbm.readthedocs.io/en/latest/Parameters.html
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

# args for baye_opt
T = args.t  # time horizon
Q = 1  # q-parallelism (if use analytical acq_func, q must be 1)
BO_METRIC = args.bm

# gp; includes "MA2.5", "SE", "RQ", "LR", "PO"
gp_name, gp_params = "MA2.5",{
                          "mode": "raw",      # "raw", "add", "pro" for GPs
                          "opt":"ADAM",  # opt for MLE; (quasi_newton, ADAM)
                          "epochs":128,       # epoch to run, if chosen ADAM
                          "lr":1e-1,          # learning rate for ADAM
                         }
acq_params = {
    "acq_name" : "UCB",          # acqu func; includes: "EI", "UCB", "qEI", "qUCB", "qKG"
    "N_start": 32,               # number of starts for multi-start SGA
    "raw_samples" :512,          # heuristic initialisation
    "N_MC_sample" : 256,         # number of samples for Monte Carlo simulation
    "num_fantasies": 128,        # number of fantasies used by KG
    "beta":1.,                   # used by UCB/qUCB
               }
"""end of Args"""

# ------------------------------------------------------------------------------------------
# median_pc_diff = (rateamount_mean - competitor_median_rate) / competitor_median_rate
# totalrevenue = room revenue + other revenue;
# Average Daily Rate (ADR) = revenue/rooms

df, data_dict, preds, ts = preprocessing(DIR, os.path.join(DIR, "data", "hotel-4_12jan2021.csv"),  \
                YEAR, DATA_RANGE, HISTORY, NDAYS_AHED, TARGET, N_CLUSTER, \
                ALL_FEAT, LAG_FEAT, LAG_DAYS, ROLLING_WINDOWS,\
                INTER_FEAT, INTER_METHODS)

"""
bayes optimisation

    notice tree model is in-variant to scale so no normalisation
"""


data_files = os.listdir(os.path.join(DIR, "data", "log"))
if "optimal_config.npy" in data_files:
    print(f"{bcolors.FAIL}optimal config already exists !! {bcolors.ENDC}")

else:
    print("---------------------------------------------")
    print(f"{bcolors.HEADER}starts bayes_opt: {bcolors.ENDC}")
    print(f"{bcolors.INFO_CYAN}data_dict len: {len(list(data_dict.keys()))} {bcolors.ENDC}")

    xs, ys = [], []
    for name in ["xgb", "lgb"]:
        if name == "xgb":
            domain = np.array([  # -> (2, d) this will change as search variale changes
                # [0,2],
                [1e-2, 0.7],
                [1,10],
                [1,5],
                [0.3, 1],
                [1, 7],
                ]).T

            cv = cv_scores("xgb", data_dict, np.zeros_like(list(data_dict.keys()))-1, -1, xgb_params, CAT_LIST, EPOCHS, KFOLD, \
                xgb_train, xgb_predict, ts, forecast_metric, TARGET, **xgb_train_params)

        elif name == "lgb":
            domain = np.array([  # -> (2, d) this will change as search variale changes
            # [0, 3], 
            [1e-2, 0.7],  # lr cannot be too small
            [20,50],
            [0.5,1],
            [0.3, 1],
            [1, 7],
            [1, 5],
            ]).T

            cv = cv_scores("lgb", data_dict, np.zeros_like(list(data_dict.keys()))-1, -1, lgb_param, CAT_LIST, EPOCHS, KFOLD, \
                lgb_train, lgb_predict, ts, forecast_metric, TARGET, **lgb_train_param)

        bayes_opt = bayesian_optimiser(T, domain, Q, gp_name, gp_params, acq_params)

        x, y = bayes_loop(bayes_opt, cv, df, BO_METRIC)

        xs.append(x)
        ys.append(y)

    # post-process the results
    xgb_cv = cv_scores("xgb", data_dict, np.zeros_like(list(data_dict.keys()))-1, -1, xgb_params, CAT_LIST, EPOCHS, KFOLD, \
                xgb_train, xgb_predict, ts, forecast_metric, TARGET, **xgb_train_params)

    lgb_cv = cv_scores("lgb", data_dict, np.zeros_like(list(data_dict.keys()))-1, -1, lgb_param, CAT_LIST, EPOCHS, KFOLD, \
                lgb_train, lgb_predict, ts, forecast_metric, TARGET, **lgb_train_param)

    optimal_config, xgb_df, lgb_df = BO_post_process(xs, ys, xgb_cv, lgb_cv)

    # saving
    xgb_df.to_csv(os.path.join(DIR, "data", "log", "bo_xgb.csv"))
    lgb_df.to_csv(os.path.join(DIR, "data", "log", "bo_lgb.csv"))

    np.save(os.path.join(DIR, "data", "log", "optimal_config.npy"), optimal_config)
    print(f"{bcolors.HEADER}done bayes_opt for optimal config {bcolors.ENDC}")
