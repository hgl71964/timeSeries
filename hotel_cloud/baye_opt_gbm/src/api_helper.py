import numpy as np
import torch as tr
import copy
from time import sleep
import os
import concurrent.futures
import multiprocessing
import asyncio

class api_utils:

    @staticmethod
    def query_wrapper(x):

        return x

    @staticmethod
    def transform(api_func: callable, metric: str):
        """
        wrap the api service;
            api_func acts on cpu, while bayes_opt at GPU
        """

        def wrapper(*args, **kwargs):
            """
            Returns:
                neg_margins: [q, 1]
            """
            x = x.cpu()
            q = x.shape[0]  # should be 1 for now 
            neg_rewards = tr.zeros(q, )
            
            x = api_utils.query_wrapper(x)

            for _ in range(5): 
                try:
                    for i in range(q):  
                        softdtw, mse = api_func(*args, **kwargs)  

                        if metric == "softdtw":
                            score = 
                        elif metric == "mse":
                            score =

                        neg_rewards[i] = -(r/r0)   # record normalised negative margin

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper    
