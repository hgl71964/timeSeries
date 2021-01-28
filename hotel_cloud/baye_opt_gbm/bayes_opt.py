import torch as tr
import numpy as np
from pandas import DataFrame
import botorch
import gpytorch
from typing import List

try:  # __main__
    from src import GPs  #  this script should be imported as packages
    from src.api_helper import api_utils

except ImportError:  # import as package
    from .src import GPs
    from .src.api_helper import api_utils

dtype = tr.float32


def BO_post_process(xs: List[tr.Tensor], 
                    ys: List[tr.Tensor],
                    xgb_cv: object,
                    lgb_cv: object,
                    ):

    """
    post-process BO results; pick optimal configuration 
    """    

    if ys[1].max() >= ys[0].max():
        name = "lgb"
        optimal_config = xs[1][ys[1].argmax()]
    else:
        name = "xgb"
        optimal_config = xs[0][ys[0].argmax()]

    if isinstance(optimal_config, tr.Tensor):
        optimal_config = optimal_config.numpy()
    
    if name == "lgb":
        optimal_config = lgb_cv.numeric_to_dict(optimal_config)
        optimal_config["name"] = name
    elif name == "xgb":
        optimal_config = xgb_cv.numeric_to_dict(optimal_config)
        optimal_config["name"] = name

    xgbs, lgbs = [], []
    for i, item in enumerate(xs[0]):
        xgb_param = xgb_cv.numeric_to_dict(np.array(item))
        xgb_param["name"] = "xgb"
        xgb_param["score"] = float(ys[0][i])
        xgbs.append(xgb_param)
    
    for i, item in enumerate(xs[1]):
        lgb_param = lgb_cv.numeric_to_dict(np.array(item))
        lgb_param["name"] = "lgb"
        lgb_param["score"] = float(ys[1][i])
        lgbs.append(lgb_param)

    return optimal_config, DataFrame(xgbs), DataFrame(lgbs)


def bayes_loop(bayes_opt: object,
                cv: object, 
                df: DataFrame,
                metric_name: str, 
                device: tr.device = tr.device("cpu"), 
                ):
    """
    func:
        cv.run_cv(df) -> (a, b)
    """

    x0 = cv.dict_to_numeric
    y0 = api_utils.init_reward(cv, df, metric_name)

    print(f"{bcolors.INFO_CYAN}initial x, y: \n ({x0}, {-y0:.2f}) {bcolors.ENDC}")

    #  format the initial pair
    x0, y0 = tr.tensor(x0, dtype=dtype).view(1, -1).to(device), \
            tr.tensor([y0],dtype=dtype).view(1, -1).to(device)

    #  decorate the api
    api = api_utils.api_wrapper(cv, metric_name)

    return bayes_opt.outer_loop(df, x0, y0, api)


class bayesian_optimiser:
    """
    data type assume torch.tensor.float() (float32)
    the optimiser is set to MAXIMISE function!
    """
    def __init__(self, 
                T: int, 
                domain: np.ndarray, # shape(2, d) [[min, max].T, ... ]
                batch_size: int,  
                gp_name: str, 
                gp_params: dict, 
                params: dict,  #  params for acquisition functions
                device: tr.device = tr.device("cpu"),
                ):
        self.T = T
        self.batch_size = batch_size
        self.gpr = self._init_GPs(gp_name, gp_params, device)  #  instantiate GP
        self.device = device
        self.params = params  # acqu_func 
        self.bounds = tr.from_numpy(domain).float().to(self.device)

    def outer_loop(self, 
                    df: DataFrame, 
                    x: tr.Tensor, # init samples; [n,d] -> n samples, d-dimensional
                    y: tr.Tensor, # shape shape [n,1]; 1-dimensional output
                    api: callable, 
                    ):
        mll, model = self.gpr.init_model(x, y, state_dict=None)

        for t in range(self.T):

            # fit model every round
            self.gpr.fit_model(mll, model, x, y)

            # acquisition function && query
            acq = self._init_acqu_func(model, y)
            query = self._inner_loop(acq, self.batch_size, self.bounds)

            reward = api(df, query).to(self.device)

            # append available data && update model
            x, y = tr.cat([x, query]), tr.cat([y, reward])
            mll, model = self.gpr.init_model(x, y, state_dict=model.state_dict())

            print(f"{bcolors.OKGREEN}Iter: {t+1}, reward: {-(reward.max()).item():,.2f}{bcolors.ENDC}")
        
        return x, y

    def _inner_loop(self, acq_func, batch_size, bounds):
        candidates, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=self.params["N_start"],       # number of starting point SGD
        raw_samples=self.params["raw_samples"],    # heuristic init
        sequential=False,                        # this enable SGD, instead of one-step optimal
        )
        query = candidates.detach()  # remove the variable from computational graph
        return query

    def _init_acqu_func(self,model,ys):
        if self.params["acq_name"] =="qKG":
            """
            if use sampler && resampling then slower
            """
            # sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.qKnowledgeGradient(
                model=model,
                num_fantasies=self.params["num_fantasies"],
                # sampler=sampler,
                objective=None,
            )
        elif self.params["acq_name"] == "qEI":
            sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.monte_carlo.qExpectedImprovement(
                model=model,
                best_f=ys.max(),
                sampler=sampler,
                objective=None, # identity objective; potentially useful model with constraints
            )
        elif self.params["acq_name"] == "qUCB":
            sampler = self._init_MCsampler(num_samples = self.params["N_MC_sample"])
            acq = botorch.acquisition.monte_carlo.qUpperConfidenceBound(
                model=model,
                beta=self.params["beta"],
                sampler=sampler,
                objective=None,
            )
        elif self.params["acq_name"] == "EI":
            acq = botorch.acquisition.analytic.ExpectedImprovement(
                model=model,
                best_f=ys.max(),
                objective=None,
            )
        elif self.params["acq_name"] == "UCB":
            acq = botorch.acquisition.analytic.UpperConfidenceBound(
                model = model,
                beta = self.params["beta"],
                objective = None,
            )
        # elif self.params["acq_name"] == "MES":
        #     acq = botorch.acquisition.max_value_entropy_search.qMaxValueEntropy(
        #         model=model,
        #         candidate_set=torch.rand(self.params["candidate_set"]),
        #         num_fantasies=self.params["MES_num_fantasies"], 
        #         num_mv_samples=10, 
        #         num_y_samples=128,            
        #         )
        
        return acq

    def _init_MCsampler(self,num_samples):
        return botorch.sampling.samplers.SobolQMCNormalSampler(num_samples=num_samples)

    def _init_GPs(self,gp_name,gp_params, device):
        return GPs.BOtorch_GP(gp_name, device, **gp_params)




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    INFO_CYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'