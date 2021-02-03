import numpy as np
import pandas as pd 
import datetime
from glob2 import glob
import os 


class helper:
    
    @staticmethod
    def _CV_average_over_folds(df):
        df["max_of_maxes"] = df.max()["max"]
        df["min_of_mines"] = df.min()["min"]
        df["mean_of_means"] = df.mean()["mean"]
        df = df.drop(columns=["min", "max", "mean"]).iloc[0]
        return df
    
    @staticmethod
    def _CV_add_weighted_mean(df, num_groups, metric_name):

        size_recorder, mean_recorder = [], []        
        temp_df = df[df["metric"]==f"{metric_name}"]
        for i in range(num_groups):

            size_recorder.append(   \
                    float(temp_df[temp_df["group_label"]==i]["group_size"].to_numpy()))
            mean_recorder.append(   \
                    float(temp_df[temp_df["group_label"]==i]["mean_of_means"].to_numpy()))

        size_recorder, mean_recorder = np.array(size_recorder), np.array(mean_recorder)
        fraction = size_recorder/size_recorder.sum()
        df[f"{metric_name}_weighted_mean"] = np.multiply(fraction, mean_recorder).sum()
        return df

    @staticmethod
    def CV_post_process(num_groups: int,  # numbers of group in total
                    *args):

        """
        make sure feature names are consistent (do not rename existing feature...)
        """
        softdtw_df, mse_df = [], []

        for df in args:

            # add stats over folds, -> pd.series
            series = helper._CV_average_over_folds(df)

            if series["metric"] == "softdtw":
                softdtw_df.append(series)
            elif series["metric"] == "mse":
                mse_df.append(series)

        if len(softdtw_df) == 1:  # on the unclustered data, i.e. group_num = -1
            return pd.concat(softdtw_df+mse_df, axis=1).T
        else:
            df1 = helper._CV_add_weighted_mean(pd.concat(softdtw_df,axis=1).T, num_groups, "softdtw")
            df2 = helper._CV_add_weighted_mean(pd.concat(mse_df,axis=1).T, num_groups, "mse")
            return pd.concat([df1, df2], axis=0)

    @staticmethod
    def feature_important(bst, name):
        if name == "xgb":
            scores = bst.get_score(importance_type="weight")
            return sorted([(key, val) for key, val in scores.items()], key=lambda x:x[-1], reverse=True)
        elif name == "lgb":
            names = bst.feature_name()
            scores = bst.feature_importance(importance_type="split")
            return sorted([(i, j) for i,j in zip(names, scores) ], key=lambda x:x[-1], reverse=True)

class logger:

    @staticmethod
    def save_df(index: int, name:str, params: dict, *args):

        if "name" not in params:
            params["name"] = name  # e.g. xgboost
        
        zidx = str(index).zfill(3)  # 3 figs naming

        # TODO the file structure has changed!
        file_present = glob(f"./data/log/{zidx}_param.csv") or \
                    glob(f"./data/log/{zidx}_metric.csv")
        
        if file_present:
            raise FileExistsError(f"file No. {zidx} exists!")
        else:
            """save params"""
            temp = pd.DataFrame(list(params.items())).T
            header = temp.iloc[0]
            temp = temp[1:]
            temp.columns = header
            pd.DataFrame(temp).to_csv(f"./data/log/{zidx}_param.csv")  

            """save cv logs"""
            if isinstance(args[0], pd.Series):
                pd.concat(args, axis=1).T.reset_index(drop=True).to_csv(f"./data/log/{zidx}_metric.csv")
            elif isinstance(args[0], pd.DataFrame):
                pd.concat(args, axis=0).to_csv(f"./data/log/{zidx}_metric.csv")
            
            print("save complete")
        return None

    @staticmethod
    def show_df(file_path: str):
        return None


    @staticmethod
    def show_all_df(dir_path: str,  # path to the directory  
                        ):

        files = os.listdir(dir_path)

        param_df, metric_df = [], []

        for f in sorted(files, key=lambda x:x[:3]):  # 3 fig naming
            full_path = os.path.join(dir_path, f)

            if "param" in f:
                param_df.append(pd.read_csv(full_path))
            elif "metric" in f:
                metric_df.append(pd.read_csv(full_path))

        return pd.concat(param_df, axis=0), pd.concat(metric_df, axis=0).reset_index(drop=True).drop(columns=["Unnamed: 0"])


class plotter:
    pass