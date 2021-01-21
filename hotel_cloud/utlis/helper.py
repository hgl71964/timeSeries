import numpy as np
import pandas as pd 
import datetime
from typing import List
from glob2 import glob

class logger:

    @staticmethod
    def save_df(index: int, name:str, params: dict, *args):

        if "name" not in params:
            params["name"] = name  # e.g. xgboost

        file_present = glob(f"./data/log/param_{index}.csv") or \
                    glob(f"./data/log/metric_{index}.csv")
        
        if file_present:
            raise FileExistsError(f"file No. {index} exists!")
        else:
            pd.DataFrame(params, index=[0]).to_csv(f"./data/log/param_{index}.csv")  
            pd.concat(args, axis=1).to_csv(f"./data/log/metric_{index}.csv")
            print("save complete")
        return None

    @staticmethod
    def show_df(file_path: str):
        return None


    @staticmethod
    def show_all_df(path: str,  # path to the directory  
                        ):

        # TODO open all file (df) and concat the results    

        logger.show_df()

        return None