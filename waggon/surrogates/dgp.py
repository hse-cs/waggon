import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
# from wgpot import Wasserstein_GP
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
import gpytorch.mlls
from contextlib import redirect_stdout
import numpy as np

class ToyDeepGPHiddenLayer(DeepGPLayer):
    # Наследуется от базового класса скрытых слоев DeepGPLayer.

    def __init__(self, input_dims, output_dims=100, num_inducing=32, mean_type='constant'):
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

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

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

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        # Возвращает MultitaskMultivariateNormal distribution или MultivariateNormal distribution если output_dims=None. То есть возвращает Гауссовский вектор - многомерное нормальное распределение.
        return super().__call__(x, are_samples=bool(len(other_inputs)))
    

num_hidden_dims = 2


class DGP(DeepGP):
    # Модуль наследуется от DeepGP - контейнера. Он в свою очередь (в итоге) наследуется от nn.Module - базового класса Pytorch для нейронных сетей.
    def __init__(self, input_dim, out_dim, num_samples=1, epochs=1500):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=out_dim,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='ZeroMean',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.epochs = epochs
        self.num_samples=num_samples

    def forward(self, inputs):
        # Классическое прохождение входящих переменных через слои сети
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def fit(self, X, y):
      
      Xy_train = TensorDataset(torch.tensor(X).double(), torch.tensor(y).double())
      train_loader = DataLoader(Xy_train, batch_size=16, shuffle=True)

      self.zero_grad()
      num_epochs = self.epochs
      num_samples = self.num_samples
      self.loss_history = []

      optimizer = torch.optim.Adam([
          {'params': self.parameters()},
      ], lr=0.01)
      mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))

      epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
      minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
      for i in epochs_iter:
        for x_batch, y_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                output = self(x_batch)
                loss = -mll(output, y_batch)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
        self.loss_history.append(loss.item())
      return self.loss_history

    def predict(self, x):
      x = torch.tensor(x)
      self.eval()
      with torch.no_grad():
          mus = []
          variances = []

          preds = self(x).detach()
          #preds = self.likelihood(self(x)) # uncomment to get likelihoods of predictions
          mus.append(preds.mean.detach())
          variances.append(preds.variance.detach())
      self.train()
      return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """

        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        # Метод call идет из базового класса Module, по умолчанию передает содержимое в forward, затем проверяет, чтобы тип выходной переменной был один из этих - (Distribution, torch.Tensor, LinearOperator)
        # Принимает значения размерности от двух
        return super().__call__(x)
    

# Возвращает предсказание на каждой эпохе обучения

class DGPSimpleMean(DGP):


    def __init__(self, input_dim, out_dim, num_samples=1, epochs=1500):
        super().__init__(input_dim, out_dim, num_samples, epochs)


    def fit(self, train_mean_x : torch.Tensor, train_y_mean : torch.Tensor):
      num_epochs = self.epochs
      num_samples = self.num_samples
      self.loss_history = []
      self.predictions_history = []

      optimizer = torch.optim.Adam([
          {'params': self.parameters()},
      ], lr=0.01)
      mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))


      epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")

      for _ in epochs_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = self(train_mean_x)
            loss = -mll(output, train_y_mean)
            loss.backward()
            loss = loss.item()
            optimizer.step()
        self.loss_history.append(loss)

        self.predictions_history.append(self.__predict(train_mean_x).mean().item())
      return self.loss_history
    
    def __predict(self, x : torch.Tensor):
        with torch.no_grad():
            mus = []
            variances = []

            preds = self(x)   
            #preds = self.likelihood(self(x)) # uncomment to get likelihoods of predictions
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1)

    def predict(self, x : torch.Tensor):
        # if x in self.predictions_history:
        #     return np.mean(np.array(self.predictions_history[x]))
        # else:
        return self.__predict(torch.Tensor([[x]])).mean()
    

# Distributional GP - принимает точку на вход и возвращает распределение

class DistributionalDGP(DGPSimpleMean):
    def __init__(self, input_dim, out_dim, num_samples=100, epochs=1500):
        super().__init__(input_dim, out_dim, num_samples, epochs)
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=100,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=100,
            output_dims=out_dim,
            mean_type='ZeroMean',
        )

        batch_shape = torch.Size([out_dim])

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.epochs = epochs
        self.num_samples=num_samples

    # if mean_type == 'constant':
        # self.mean_module = ConstantMean(batch_shape=batch_shape)
    # else:
        self.mean_module = LinearMean(input_dim)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def fit(self, train_mean_x : torch.Tensor, train_y_mean : torch.Tensor):
      num_epochs = self.epochs
      self.loss_history = []
      self.predictions_history = []

      optimizer = torch.optim.Adam([
          {'params': self.parameters()},
      ], lr=0.01)
      mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))


      epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")

      for _ in epochs_iter:
        with gpytorch.settings.num_likelihood_samples(self.num_samples):
            output = self(train_mean_x)
            print(output, train_y_mean.shape)
            loss = -mll(output, train_y_mean)
            print(loss)
            loss.backward(retain_graph=True)
            loss = loss.item()
            optimizer.step()

            optimizer.zero_grad()
        self.loss_history.append(loss)

        with torch.no_grad():
            self.predictions_history.append(self.forward(train_mean_x))
      return self.loss_history

    def __predict(self, x):
        return torch.tensor([1.])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        with gpytorch.settings.num_likelihood_samples(self.num_samples):
            if len(other_inputs):
                if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                    x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    for inp in other_inputs
                ]

                x = torch.cat([x] + processed_inputs, dim=-1)

            # Метод call идет из базового класса Module, по умолчанию передает содержимое в forward, затем проверяет, чтобы тип выходной переменной был один из этих - (Distribution, torch.Tensor, LinearOperator)
            # Принимает значения размерности от двух
        return super().__call__(x)


class WassersteinDGP(DistributionalDGP):
    def __init__(self, input_dim, out_dim, num_samples=100, epochs=1500):
        super().__init__(input_dim, out_dim, num_samples, epochs)
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=out_dim,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=out_dim,
            mean_type='ZeroMean',
        )

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.epochs = epochs
        self.num_samples=num_samples

    def fit(self, train_mean_x : torch.Tensor, train_y_mean : torch.Tensor):
      num_epochs = self.epochs
    #   num_samples = self.num_samples
      self.loss_history = []
      self.predictions_history = []

      optimizer = torch.optim.Adam([
          {'params': self.parameters()},
      ], lr=0.01)
      mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))


      epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")

      for _ in epochs_iter:
        with gpytorch.settings.num_likelihood_samples(self.num_samples):
            optimizer.zero_grad()
            output = self(train_mean_x)
            print(output, train_y_mean.shape)
            loss = -mll(output.mean[0], train_y_mean)
            print(loss.shape)
            loss.backward()
            loss = loss.item()
            optimizer.step()
        self.loss_history.append(loss)

        self.predictions_history.append(self.__predict(train_mean_x).mean().item())
      return self.loss_history
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        with gpytorch.settings.num_likelihood_samples(self.num_samples):
            if len(other_inputs):
                if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                    x = x.rsample()

                processed_inputs = [
                    inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                    for inp in other_inputs
                ]

                x = torch.cat([x] + processed_inputs, dim=-1)

            # Метод call идет из базового класса Module, по умолчанию передает содержимое в forward, затем проверяет, чтобы тип выходной переменной был один из этих - (Distribution, torch.Tensor, LinearOperator)
            # Принимает значения размерности от двух
            return super().__call__(x)


    def sample(x : torch.Tensor):
        return x

    def predict(self, x : torch.Tensor):
        return np.sample()
#     def Barycenter(self, dot_x, dot_y):
        # gp_0 = (mu_0, k_0)     
        # gp_1 = (mu_1, k_1)
        # # mu_0/mu_1 (ndarray (n, 1)) is the mean of one Gaussian Process 
        # # K_0/K_1 (ndarray (n, n)) is the covariance matrix of one 
        # # Gaussain Process

        # wd_gp = Wasserstein_GP(gp_0, gp_1)











# from .surr import Surrogate

# import torch
# import numpy as np
# from tqdm import tqdm
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# import gpytorch
# import gpytorch.mlls
# from gpytorch.means import LinearMean
# from gpytorch.mlls import VariationalELBO
# from gpytorch.mlls import DeepApproximateMLL
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.kernels import RBFKernel, ScaleKernel
# from gpytorch.distributions import MultivariateNormal
# from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
# from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution

# # cuda = True if torch.cuda.is_available() else False
# Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else


# class DGPLayer(DeepGPLayer):

#     def __init__(self, input_dims, output_dims=None, num_inducing=32):
#         if output_dims is None:
#             inducing_points = torch.randn(num_inducing, input_dims)
#             batch_shape = torch.Size([])
#         else:
#             inducing_points = torch.randn(output_dims, num_inducing, input_dims)
#             batch_shape = torch.Size([output_dims])

#         variational_distribution = CholeskyVariationalDistribution(
#             num_inducing_points=num_inducing,
#             batch_shape=batch_shape
#         )

#         variational_strategy = VariationalStrategy(
#             self,
#             inducing_points,
#             variational_distribution,
#             learn_inducing_locations=True
#         )

#         super(DGPLayer, self).__init__(variational_strategy, input_dims, output_dims)

#         self.mean_module = LinearMean(input_dims)

#         self.covar_module = ScaleKernel(
#             RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
#             batch_shape=batch_shape, ard_num_dims=None
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return MultivariateNormal(mean_x, covar_x)

#     def __call__(self, x, *other_inputs):
#         if len(other_inputs):
#             if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
#                 x = x.rsample()

#             processed_inputs = [
#                 inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
#                 for inp in other_inputs
#             ]

#             x = torch.cat([x] + processed_inputs, dim=-1)
#         return super().__call__(x, are_samples=bool(len(other_inputs)))


# class DGP(DeepGP):
#     def __init__(self, **kwargs):
#         super(DGP, self).__init__()

#         self.name        = 'DGP'
#         self.model       = kwargs['model'] if 'model' in kwargs else None
#         self.n_epochs    = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
#         self.lr          = kwargs['lr'] if 'lr' in kwargs else 1e-3
#         self.batch_size  = kwargs['batch_size'] if 'batch_size' in kwargs else 16
#         self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 10
#         self.n_samples   = kwargs['n_samples'] if 'n_samples' in kwargs else 1
#         self.save_loss   = kwargs['save_loss'] if 'save_loss' in kwargs else False
#         self.verbose     = kwargs['verbose'] if 'verbose' in kwargs else 1

#         self.likelihood  = GaussianLikelihood()

#         if self.save_loss:
#             self.loss_hist = []

#     def forward(self, x):
#         x = self.model(x)
#         return x

#     def fit(self, X, y):

#         if self.model is None:
#             self.model = nn.Sequential(
#                 DGPLayer(input_dims=X.shape[-1], output_dims=self.hidden_size),
#                 DGPLayer(input_dims=self.hidden_size)
#             )
        
#         X = Tensor(X)
#         y = Tensor(y)
#         X_train = TensorDataset(X, y)
#         self.zero_grad()

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#         mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, 10))

#         self.train()

#         fit_loop = range(self.n_epochs)
#         if self.verbose > 1:
#             fit_loop = tqdm(fit_loop, desc="epochs")
        
#         for e in fit_loop:
#             for X_batch, y_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):
                
#                 with gpytorch.settings.num_likelihood_samples(self.n_samples):
                    
#                     y_pred = self(X_batch)
#                     loss = -mll(y_pred, y_batch)
                    
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
                
#                 if self.save_loss:
#                     self.loss_hist.append(loss.item())
#         self.eval()

#     def predict(self, X):

#         X = Tensor(X)
        
#         with torch.no_grad():
#             with gpytorch.settings.num_likelihood_samples(self.n_samples):
#                 preds = self(X)
#                 f = preds.mean.detach().numpy()
#                 std = np.sqrt(preds.variance.detach().numpy())
        
#         return f, std

#     def __call__(self, x, *other_inputs):
#         if len(other_inputs):
#             if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
#                 x = x.rsample()

#             processed_inputs = [
#                 inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
#                 for inp in other_inputs
#             ]

#             x = torch.cat([x] + processed_inputs, dim=-1)
#         return super().__call__(x)
