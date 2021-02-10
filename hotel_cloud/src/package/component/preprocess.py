"""
this script runs bayes_opt to find optimal hyperparameter for (xg, light)gbm models

the results are store -> ${work_dir}/data/optimal_config.npy
logs are store -> ${work_dir}/data/log/bo*
"""
import os 
import numpy as np
import pandas as pd
from ..utlis.color import bcolors
from typing import List

"""data cleansing"""
from ..utlis.timeSeries_dataset import timeSeries_data

"""clustering"""
from tslearn.clustering import TimeSeriesKMeans
from ..models.kMeansTimeSeries import Kmeans_predict

def preprocessing(working_dir: str,  # path of the working dir
                data_full_path: str,  # path to get original dataset
                year,  # only for check
                data_range: tuple,
                ndays_ahead: int,
                target: str,
                n_cluster: int,
                all_feats: List[str],  # all feats involved in modelling
                lag_feats: List[str],  # the feats that need to make lag
                lag_days: List[int],
                rolling_feats: List[str],  # features to be applied to rolling
                rolling_windows: List[int],  # windows for rolling features
                ):
    """
    takes in raw data,
        return cleansed, lagged features, rolling features dataframe

    also perform clustering in this function, return group label
                    and a dict index -> date
    """

    raw_df = pd.read_csv(data_full_path)
    raw_df["reportdate"] = raw_df["reportdate"].astype("datetime64[ns]")
    raw_df["staydate"] = raw_df["staydate"].astype("datetime64[ns]")

    """ data cleansing && add lag features """
    
    print(f"{bcolors.INFO_CYAN}start data cleansing {bcolors.ENDC}")

    ts = timeSeries_data(**{"year": year, })

    # TODO add rolling 
    data, data_dict, df = ts.cleansing(raw_df, all_feats, data_range, target, \
                        ndays_ahead, lag_feats, lag_days, **{"interpolate_col": all_feats})

    # ----------------------------------------------------------------------------------------
    """ clustering """
    data_files = os.listdir(os.path.join(working_dir, "data", "log"))

    if "preds.npy" in data_files:
        print(f"{bcolors.HEADER}reading from data folder... {bcolors.ENDC}")
        preds = np.load(os.path.join(working_dir, "data", "log","preds.npy"))

    else:
        # euclidean, softdtw, dtw
        print(f"{bcolors.INFO_CYAN}labels does not exist; start clustering... {bcolors.ENDC}")
        _, preds = Kmeans_predict(data, n_cluster, **{"metric": "softdtw"})

        # save clustering results
        np.save(os.path.join(working_dir, "data", "log", "preds.npy"), preds)
        print(f"{bcolors.WARNING}done saving {bcolors.ENDC}")
    
    return (df,
            data_dict,
            preds,
            ts)