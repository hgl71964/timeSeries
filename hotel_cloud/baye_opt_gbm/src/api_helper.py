import numpy as np
import torch as tr
from time import sleep
import os
import asyncio

class api_utils:

    @staticmethod
    def query_wrapper(x):
        return x

    @staticmethod
    def api_wrapper(api_func: callable, metric: str):

        def wrapper(df,
                    x,  # query in numeric
                    device, 
                    ):  
            x = x.cpu()
            q = x.shape[0]  # should be 1 for now 
            neg_rewards = tr.zeros(q, )
            
            # pre-process api
            x = api_utils.query_wrapper(x)

            for _ in range(5): 
                try:
                    for i in range(q):  
                        softdtw, mse = api_func(*args, **kwargs)  

                        if metric == "softdtw":
                            score = softdtw["mean"].mean()
                        elif metric == "mse":
                            score = mse["mean"].mean()

                        neg_rewards[i] = - score   # record normalised negative margin

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper    
