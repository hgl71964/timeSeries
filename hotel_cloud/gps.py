import torch as tr
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class gp_model2:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forecast_len: int, # length that need to forecast
                model=None,  # a gpytorch model
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

        if forecast_len >= n:  # check forecast_len
            raise ValueError("history too short")

        self.arr = tr.from_numpy(self.arr.astype(np.float32))  # to Tensor

        # make end point observation points
        self.train_y = tr.cat([self.arr[:n - forecast_len], tr.tensor([self.arr[-1]])]).float()
        self.train_x = tr.cat([tr.arange(0, len(self.train_y)-1), tr.tensor([len(self.arr)-1])]).float()

        # zero noise likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        self.likelihood.noise = 1e-4
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)

        if model is None:
            self.model = SpectralMixtureGPModel(self.train_x, self.train_y, self.likelihood)
        else:
            self.model = model(self.train_x, self.train_y, self.likelihood)

    def train(self, verbose=True, **kwargs):
        lr = kwargs.get("lr", 1e-1)
        epochs = kwargs.get("epoch", 128)

        self.model.train()
        self.likelihood.train()
        optimizer = tr.optim.Adam(self.model.parameters(), lr=lr)  

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(epochs):

            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            
            if verbose:
                print(f"Iter {i+1}/{epochs} - Loss {loss.item():.3f} - Noise {self.model.likelihood.noise.item():.3f}")
            optimizer.step()

    @property
    def plot_prediction(self):

        self.model.eval(); self.likelihood.eval()

        with tr.no_grad():

            test_x = tr.arange(0, len(self.arr)).float()
            y_preds = self.likelihood(self.model(test_x))
            lower, upper = y_preds.confidence_region()

            fig, ax = plt.subplots()
            # ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
            ax.plot([i for i in range(len(self.arr))] , self.arr, color="black")
            ax.plot(test_x.numpy(), y_preds.mean.numpy(), 'blue')
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.grid(True)

            ax.legend(["objective", "Mean", "Confidence"])
            plt.show()

        return None

class gp_model:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forecast_len: int, # length that need to forecast
                model=None,  # a gpytorch model
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

        if forecast_len >= n:  # check forecast_len
            raise ValueError("history too short")

        self.arr = tr.from_numpy(self.arr.astype(np.float32))  # to Tensor

        self.train_y = self.arr[:n - forecast_len].float()
        self.train_x = tr.arange(0, len(self.train_y)).float()

        print(self.train_x.shape, self.train_y.shape)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        if model is None:
            self.model = SpectralMixtureGPModel(self.train_x, self.train_y, self.likelihood)
        else:
            self.model = model(self.train_x, self.train_y, self.likelihood)

    def train(self, verbose=True, **kwargs):
        lr = kwargs.get("lr", 1e-1)
        epochs = kwargs.get("epoch", 128)

        self.model.train()
        self.likelihood.train()
        optimizer = tr.optim.Adam(self.model.parameters(), lr=lr)  

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(epochs):

            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            
            if verbose:
                print(f"Iter {i+1}/{epochs} - Loss {loss.item():.3f} - Noise {self.model.likelihood.noise.item():.3f}")
            optimizer.step()

    @property
    def plot_prediction(self):

        self.model.eval(); self.likelihood.eval()

        with tr.no_grad():

            test_x = tr.arange(0, len(self.arr)).float()
            y_preds = self.likelihood(self.model(test_x))
            lower, upper = y_preds.confidence_region()

            fig, ax = plt.subplots()
            # ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
            ax.plot([i for i in range(len(self.arr))] , self.arr, color="black")
            ax.plot(test_x.numpy(), y_preds.mean.numpy(), 'blue')
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.grid(True)

            ax.legend(["objective", "Mean", "Confidence"])
            plt.show()

        return None

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=2)
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




if __name__ == "__main__":
    a = np.random.rand(100,)

    gp_model2(a, 60 ,10 )