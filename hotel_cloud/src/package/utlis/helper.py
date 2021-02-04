import numpy as np
import pandas as pd 
from typing import List
import datetime
from glob2 import glob
import os 


class helper:

    @staticmethod
    def worst_day_res(test_dates: List[str],
                        df,  
                        ts: object,                        
                        predict_func: callable,
                        cat_list, 
                        target, 
                        bst,  # trained bst
                        metric_name: str, 
                        metric: object, 
                        ):
        res = []
        for test_date in test_dates:
            ivd_test_df =  ts.dataset_from_dates(df, [test_date])
            preds = predict_func(ivd_test_df, cat_list, target, bst)

            if metric_name == "softdtw":
                temp = metric.softdtw(preds, ivd_test_df[target])
            elif metric_name == "mse":
                temp = metric.mse(preds, ivd_test_df[target])

            res.append(temp)

        maxpos = res.index(max(res))  # worst date index
        worst_date = test_dates[maxpos]
        ivd_test_df =  ts.dataset_from_dates(df, [worst_date])
        preds = predict_func(ivd_test_df, cat_list, target, bst)

        return preds, ivd_test_df[target]

    @staticmethod
    def feature_important(bst, name, cat_list):
        if name == "xgb":
            
            one_hot_feats = cat_list
            one_hot_dict = {i:0 for i in one_hot_feats}
            scores = bst.get_score(importance_type="weight")

            temp = [(key, val) for key, val in scores.items()]
            pop_index = set()

            for i, item in enumerate(temp):
                for feat in one_hot_feats:  # one-hot-feat needs to sum up
                    if item[0].startswith(feat):
                        one_hot_dict[feat]+=item[1]
                        pop_index.add(i)
                        break

            for feat in one_hot_feats:
                temp.append((feat, one_hot_dict[feat]))

            return sorted([x for i, x in enumerate(temp) if i not in pop_index], \
                                    key=lambda x: x[-1], reverse=True)

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