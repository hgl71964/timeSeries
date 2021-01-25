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
    def transform(api_func: callable):
        """
        wrap the api service;
            provide small number perturbation, type conversion etc.

            api_func acts on cpu, while bayes_opt at GPU
        """

        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    device: str,
                    ):
            """
            Returns:
                neg_margins: [q, 1]
            """
            x = x.cpu(); q = x.shape[0]; neg_margins = tr.zeros(q, )
            
            # we may want to push query off the boundary
            # for i in x:
                # if np.equal(i.all(), 1.):  # very extreme case; has been tested
                        # i -= 1e-3     
            ## generally, slightly push variables off boundary         
            # x[x == 1] -= 1e-6
            # x[x == 0] += 1e-6

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    for i in range(q):  # sequential query 
                        r = api_func(x[i])  # float
                        neg_margins[i] = -(r/r0)   # record normalised negative margin

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper    


    @staticmethod
    def multi_process_transform(api_func: callable):
        """
        for cpu bound problem
        """
        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    device: str,
                    ):
            """
            multiprocessing must pickle things to sling them among processes

            Returns:
                neg_margins: [q, 1]
            """
            x = x.cpu(); q = x.shape[0]; neg_margins = tr.zeros((q, ))

            for _ in range(5):  # exception control
                try:
                    with multiprocessing.Pool(processes=10) as pool:
                        for i, r in enumerate(pool.map(api_func, x)):
                            neg_margins[i] = -(r/r0)   

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).to(device) 

        return wrapper


    @staticmethod
    def multi_thread_transform(api_func: callable):
        """
        IO bound version
        """
        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    device: str,
                    ):
            """
            Returns:
                neg_margins: [q, 1]
            """
            x = x.cpu(); q = x.shape[0]; neg_margins = tr.zeros((q, ))

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        for i, r in enumerate(executor.map(api_func, x)):  # multi-threading
                            neg_margins[i] = -(r/r0)   

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_margins.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper

    
class env:
    """
    an simulated environment; upon receiving query, give reward
    """
    @staticmethod
    def rosenbrock(query: tr.tensor):
        """
        the rosenbrock function; f(x,y) = (a-x)^2 + b(y - x^2)^2
        Global minimum: 0; at (a, a^2)
        usually a = 1, b = 100
        """
        x, y = query.flatten()  # only take as input 2-element tensor
        return tr.tensor([(1 - x)**2 + 100 * (y - x**2)**2])
    
    @staticmethod
    def rosenbrock_grad(query: tr.tensor):
        x, y = query.flatten()  # only take as input 2-element tensor 
        return tr.tensor([2+400*(x**2 - y) + 800*(x**2), 200*(y-(x**2))])
