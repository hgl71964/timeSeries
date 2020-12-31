import torch as tr
import gpytorch
import numpy as np
import pandas as pd


class gp_model:

    def __init__(self,
                arr,  # 1d-array_like
                history: int,  # total lenght of history 
                forecast_len: int, # length that need to forecast
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

        # TODO figure out train test
        self.train_y = self.arr[:n - forecast_len]
        self.train_x = tr.arange(0, len(self.train_y))

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        self.model = SpectralMixtureGPModel(self.train_x, self.train_y, self.likelihood)

    def train(self, **kwargs):
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

            print(f"Iter {i+1}/{epochs} - Loss {loss.item():.3f} - Noise {self.model.likelihood.noise.item():.3f}")
            optimizer.step()

    @property
    def plot_prediction(self):

        self.model.eval(); self.likelihood.eval()

        with tr.no_grad():

            test_x = tr.arange(len(self.train_y), len(self.arr))
            y_preds = self.likelihood(self.model(test_x))
            lower, upper = y_preds.confidence_region()

            fig, ax = plt.subplots()

            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            ax.plot(test_x.numpy(), y_preds.mean.numpy(), 'b')
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()

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

