from .base import Surrogate

import torch
import gpytorch
from tqdm import tqdm
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.models import AbstractVariationalGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class DGP(Surrogate):
    def __init__(self, **kwargs):
        super(DGP, self).__init__()

        self.name         = 'DGP'
        self.model        = kwargs['model'] if 'model' in kwargs else None
        self.n_epochs     = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.lr           = kwargs['lr'] if 'lr' in kwargs else 1e-1
        self.verbose      = kwargs['verbose'] if 'verbose' in kwargs else 1
        self.num_inducing = kwargs['num_inducing'] if 'num_inducing' in kwargs else 32
        self.hidden_size  = kwargs['hidden_size'] if 'hidden_size' in kwargs else 16
    
    def fit(self, X, y):

        if self.model is None:
            self.model = DeepGPModel(X.shape[1], y.shape[1],
                                     hidden_size=self.hidden_size,
                                     num_inducing=self.num_inducing)
        
        X = torch.tensor(X).float()
        y = torch.tensor(y).float().squeeze()
        
        self.model.train()
        self.model.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        mll = DeepApproximateMLL(VariationalELBO(self.model.likelihood, self.model, num_data=y.shape[0]))

        if self.verbose > 1:
            pbar = tqdm(range(self.n_epochs), leave=False)
        else:
            pbar = range(self.n_epochs)
        
        for epoch in pbar:
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.mean().backward()
            optimizer.step()
            
            if self.verbose > 1:
                pbar.set_description(f'Epoch {epoch + 1}/{self.n_epochs} - Loss: {loss.mean().item():.3f}')

    
    def predict(self, X):

        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(torch.tensor(X).float()))
            mean = observed_pred.mean
            std = torch.sqrt(observed_pred.variance)
        return mean[0, 0, :].numpy(), std[0, 0, :].numpy()


class SingleLayerGP(AbstractVariationalGP):
    def __init__(self, inducing_points, mean='const'):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean() if mean == 'const' else  LinearMean(inducing_points.size(1))
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=inducing_points.size(1))) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, inducing_points, mean_type='constant'):
        if output_dims is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPModel(DeepGP):
    def __init__(self, in_dim, out_dim=None, hidden_size=16, num_inducing=22, layers=['deep', 'deep']):
        super().__init__()

        gen = torch.Generator()
        gen.manual_seed(2208060503)
        inducing_points = torch.rand(num_inducing, in_dim, generator=gen)
        output_inducing = torch.rand(num_inducing, hidden_size if layers[1]=='deep' else 1, generator=gen)

        self.input_layer = DeepLayer(in_dim, hidden_size, inducing_points) if layers[0]=='deep' else SingleLayerGP(inducing_points)
        self.output_layer =  DeepLayer(hidden_size, None, output_inducing) if layers[1]=='deep' else SingleLayerGP(output_inducing)
        
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        hidden_rep = self.input_layer(x).mean
        output = self.output_layer(hidden_rep)
        return output
