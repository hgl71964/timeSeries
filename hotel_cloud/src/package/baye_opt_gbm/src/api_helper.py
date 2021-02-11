import numpy as np
import torch as tr
from time import sleep
import os

class api_utils:

    @staticmethod
    def api_wrapper(cv: object,  
                    metric_name: str, 
                    ):
        """
        only meant to decorate a specific api_func/object, 
            such that it returns reward for a provided query
        """
        def wrapper(df,
                    x,  # query in numeric
                    ):  
            x = x.cpu().numpy()
            q = x.shape[0]  # should be 1 for now 
            neg_rewards = tr.zeros(q, )

            for _ in range(5): 
                try:
                    for i in range(q):  

                        update_dict = cv.numeric_to_dict(x[i])
                        cv.update_param(update_dict)

                        print(f"{bcolors.INFO_CYAN}config: {bcolors.ENDC}")
                        print(f"{bcolors.INFO_CYAN}", cv.param)
                        print(f"{bcolors.ENDC}")

                        softdtw, mse = cv.run_cv(df)
                        temp1, temp2 = -softdtw["mean"].mean(), -mse["mean"].mean()

                        print(f"{bcolors.OKGREEN} softdtw: {temp1:.2f}")
                        print(f"mse: {temp2:.2f} {bcolors.ENDC}")

                        if metric_name == "softdtw":
                            score = softdtw["mean"].mean()
                        elif metric_name == "mse":
                            score = mse["mean"].mean()

                        neg_rewards[i] = -score   # record normalised negative margin

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_rewards.view(-1, 1)  # assume dtype == torch.float() overall

        return wrapper    

    @staticmethod
    def init_reward(cv, df, metric_name):
        softdtw, mse = cv.run_cv(df)

        temp1, temp2 = -softdtw["mean"].mean(), -mse["mean"].mean()

        print(f"{bcolors.OKGREEN} softdtw {temp1:.2f}")
        print(f"mse {temp2:.2f} {bcolors.ENDC}")

        if metric_name == "softdtw":
            return -softdtw["mean"].mean()
        elif metric_name == "mse":
            return -mse["mean"].mean()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    INFO_CYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'