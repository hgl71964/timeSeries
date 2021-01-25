import numpy as np
import torch as tr
from src.api_helper import api_utils
dtype = tr.float32


def bayes_loop(
            bayes_opt: object,
            loss_func: callable,  # decorated as api
            ):

    # get x0, y0
    x0 = api_utils.init_query()
    y0 = api_utils.init_reward()

    #  format the initial pair
    x0, y0 = tr.from_numpy(x0).to(device), y0.to(device)

    #  decorate the api
    api = api_utils.api_wrapper(loss_func, metric)

    return bayes_opt.outer_loop(x0, y0, r0, api)

