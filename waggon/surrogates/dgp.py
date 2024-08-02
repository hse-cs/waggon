# from .surr import Surrogate

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import gpytorch
import gpytorch.mlls
from gpytorch.means import LinearMean
from gpytorch.mlls import VariationalELBO
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else


class DGPLayer(DeepGPLayer):

    def __init__(self, input_dims, output_dims=None, num_inducing=32):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DGPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DGP(DeepGP):
    def __init__(self, **kwargs):
        super(DGP, self).__init__()

        self.name        = 'DGP'
        self.model       = kwargs['model'] if 'model' in kwargs else None
        self.n_epochs    = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.lr          = kwargs['lr'] if 'lr' in kwargs else 1e-3
        self.batch_size  = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 10
        self.n_samples   = kwargs['n_samples'] if 'n_samples' in kwargs else 1
        self.save_loss   = kwargs['save_loss'] if 'save_loss' in kwargs else False
        self.verbose     = kwargs['verbose'] if 'verbose' in kwargs else 1

        self.likelihood  = GaussianLikelihood()

        if self.save_loss:
            self.loss_hist = []

    def forward(self, x):
        x = self.model(x)
        return x

    def fit(self, X, y):

        if self.model is None:
            self.model = nn.Sequential(
                DGPLayer(input_dims=X.shape[-1], output_dims=self.hidden_size),
                DGPLayer(input_dims=self.hidden_size)
            )
        
        X = Tensor(X)
        y = Tensor(y)
        X_train = TensorDataset(X, y)
        self.zero_grad()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))

        self.train()

        fit_loop = range(self.n_epochs)
        if self.verbose > 1:
            fit_loop = tqdm(fit_loop, desc="epochs")
        
        for e in fit_loop:
            for X_batch, y_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):
                
                with gpytorch.settings.num_likelihood_samples(self.n_samples):
                    
                    y_pred = self(X_batch)
                    loss = -mll(y_pred, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if self.save_loss:
                    self.loss_hist.append(loss.item())
        self.eval()

    def predict(self, X):

        X = Tensor(X)
        
        with torch.no_grad():
            with gpytorch.settings.num_likelihood_samples(self.n_samples):
                preds = self(X)
                f = preds.mean.detach().numpy()
                std = np.sqrt(preds.variance.detach().numpy())
        
        return f, std

    def __call__(self, x, *other_inputs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x)
