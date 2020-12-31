import torch as tr
import gpytorch
import numpy as np
import pandas as pd


class gp:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forcast_len: int, # length that need to forecast
                device = tr.device("cpu"), 
                ):
        
        if isinstance(arr, pd.Series):  # type conversion 
            self.arr = arr.to_numpy()
        else:
            self.arr = arr

        if self.arr[-1] < self.arr[0]:  # flip arr to ascent order 
            self.arr = np.flip(self.arr)

        n = len(self.arr)
        if n > history:  #  truncate history 
            self.arr = self.arr[n-history:]
            n=len(self.arr)

        if forcast_len >= n:  # check forcast_len
            raise ValueError("history too short")

        self.arr = tr.from_numpy(self.arr.astype(np.float32))

        # TODO figure out train test
        self.train_x, self.test_x = self.arr[:forcast_len], self.arr[forcast_len:]


        self.likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        self.model = SpectralMixtureGPModel(TODO, TODO, self.likelihood)

        self.model = SpectralMixtureGPModel()

    def train(self, **kwargs):

        lr = kwargs.get("lr", 1e-1)
        epochs = kwargs.get("epoch", 128)

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(epochs):

            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.test_x)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()
    
    @property
    def forecast(self):
        return None 



class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

