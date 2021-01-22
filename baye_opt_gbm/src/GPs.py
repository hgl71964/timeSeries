import torch
import botorch
import gpytorch

class BOtorch_GP:
    """
    data type assume torch.tensor.float()

    can only apply to standard kernels GPs
    """
    def __init__(self, gp_name, device, **kwargs):
        self.name = gp_name
        self.params = kwargs
        self.device = device

    def init_model(self, x, y, state_dict=None):
        """
        this implementation of zero-noise GP is more numerically robust;
            initialise and update a BOtorch model every outside loop

        Args:
            x: training samples; shape [n, d] -> n samples, d-dimensional
            y: function values; shape [n,m] -> multi-output m-dimensional; m = 1 in our case
            state_dict: update model when it is provided

        Returns:
            mll: Gpytorch Marginal likelihood
            model: Botorch model
        """
        # zeros-noise settings
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 1e-4  
        likelihood.noise_covar.raw_noise.requires_grad_(False)

        model = botorch.models.SingleTaskGP(x, y, likelihood)

        kernel = self.make_kernel(self.name)

        if self.params["mode"] == "raw":
            model.covar_module = kernel
        elif self.params["mode"] == "add":
            model.covar_module = gpytorch.kernels.AdditiveStructureKernel(base_kernel = kernel, num_dims = x.size(-1))
        elif self.params["mode"] == "pro":
            model.covar_module = gpytorch.kernels.ProductStructureKernel(base_kernel = kernel, num_dims = x.size(-1))

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll.to(self.device), model.to(self.device)

    def fit_model(self, mll, model, x, y):
        """
        MLE tuning via ADAM
        Args:
            x -> shape[n,d]; tensor
            y -> shape[n,1]; tensor
        """
        if self.params["opt"] == "ADAM":
            self._ADAM(mll,model,x,y)
        elif self.params["opt"] == "quasi_newton":
            self._quasi_newton(mll)
    
    def make_kernel(self, name):
        if name == "SE":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif name == "RQ":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif name == "MA2.5":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        elif name == "PO2":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power = 2))
        elif name == "PO3":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power = 3))
        elif name == "LR":
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        return kernel

    def _ADAM(self, mll, model, x, y):
        """
        MLE tuning via ADAM
        Args:
            x -> shape[n,d]; tensor
            y -> shape[n,1]; tensor
        """
        # when training: y -> shape[n,]
        y  = y.squeeze(-1)

        model.train()
        model.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params["lr"])

        for _ in range(self.params["epochs"]):
            
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
     
        print(f"NLML: {loss.item():,.2f}")
        model.eval()
        model.likelihood.eval()

    def _quasi_newton(self, mll):
        """
        MLE tuning via L-BFGS-B
        mll: marginal likelihood of model; work for BOtorch and Gpytorch model

        but this cannot handle SM kernel
        """
        botorch.fit_gpytorch_model(mll)

    def __str__(self):
        return self.name