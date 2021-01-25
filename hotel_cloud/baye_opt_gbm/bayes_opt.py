import torch as tr
import numpy as np
import botorch
import gpytorch
from src import GPs  #  this script should be imported as packages
from src.api_helper import api_utils


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

    # TODO
    x0 = cv.dict_to_numeric
    y0 = api_utils.init_reward(cv, df, metric_name)

    print(x0, y0)

    #  format the initial pair
    x0, y0 = tr.from_numpy(x0).to(device), y0.to(device)

    #  decorate the api
    api = api_utils.api_wrapper(cv, metric_name)

    return bayes_opt.outer_loop(x0, y0, r0, api)





class bayesian_optimiser:
    """
    data type assume torch.tensor.float() (float32)
    the optimiser is set to MAXIMISE function!
    """
    def __init__(self, 
                T: int, 
                domain: np.ndarray, # shape(n, 2) [[min, max], ... ]
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

        # TODO refine bound
        if isinstance(domain, tuple):
            self.bounds = tr.tensor([[domain[0]] * input_dim, [domain[1]] * input_dim], \
                                        dtype=tr.float32).to(self.device)
        else:
            self.bounds = tr.from_numpy(domain).float().to(self.device)

    def outer_loop(self, 
                    x: tr.Tensor, # init samples; [n,d] -> n samples, d-dimensional
                    y: tr.Tensor, # shape shape [n,1]; 1-dimensional output
                    r0: float, # unormalised reward,
                    api: callable, 
                    ):

        input_dim = x.shape[-1]

        mll, model = self.gpr.init_model(x, y, state_dict=None)

        for t in range(self.T):

            # fit model every round
            self.gpr.fit_model(mll, model, x, y)

            # acquisition function && query
            acq = self._init_acqu_func(model, y)
            query = self._inner_loop(acq, self.batch_size, self.bounds)

            # TODO refine signature 
            reward = api(query, r0, self.device) 

            # append available data && update model
            x, y = tr.cat([x, query]), tr.cat([y, reward])
            mll, model = self.gpr.init_model(x, y, state_dict=model.state_dict())

            print(f"Iter: {t+1}, reward: {(reward.max()).item():,.2f}")
        
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
