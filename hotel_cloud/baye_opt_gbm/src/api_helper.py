import numpy as np
import torch as tr
from time import sleep
import os
import asyncio

class api_utils:

    @staticmethod
    def api_wrapper(cv: object,  
                    metric_name: str, 
                    ):
        def wrapper(df,
                    x,  # query in numeric
                    ):  
            x = x.cpu().numpy()
            q = x.shape[0]  # should be 1 for now 
            neg_rewards = tr.zeros(q, )
            
            # update params
            print("query: ")
            print(x)
            update_dict = cv.numeric_to_dict(x)

            print(update_dict)

            cv.update_param(update_dict)

            for _ in range(5): 
                try:
                    for i in range(q):  
                        softdtw, mse = cv.run_cv(df)

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
        if metric_name == "softdtw":
            return -softdtw["mean"].mean()
        elif metric_name == "mse":
            return -mse["mean"].mean()