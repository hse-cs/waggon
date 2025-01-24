from .base import Surrogate

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchbnn as bnn
from torch.utils.data import TensorDataset, DataLoader

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else


class BayesianRegressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, hidden_size=64):
        super(BayesianRegressor, self).__init__()

        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_inputs, out_features=hidden_size),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=n_outputs)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class BNN(Surrogate):# TODO: add cuda
    def __init__(self, **kwargs):
        super(BNN, self).__init__()

        self.name        = 'BNN'
        self.model       = kwargs['model'] if 'model' in kwargs else None
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 64
        self.n_epochs    = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.lr          = kwargs['G_lr'] if 'G_lr' in kwargs else 1e-3
        self.batch_size  = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        self.n_preds     = kwargs['n_preds'] if 'n_preds' in kwargs else 10
        self.verbose     = kwargs['verbose'] if 'verbose' in kwargs else 1

        self.save_loss   = kwargs['save_loss'] if 'save_loss' in kwargs else False
        self.mse_loss    = nn.MSELoss()
        self.kl_loss     = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight   = kwargs['kl_weight'] if 'kl_weight' in kwargs else 1e-2

        if self.save_loss:
            self.loss_hist = []
    
    def fit(self, X, y):

        if self.model is None:
            self.model = BayesianRegressor(n_inputs=X.shape[-1], n_outputs=y.shape[-1], hidden_size=self.hidden_size)
        
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        X = Tensor(X)
        y = Tensor(y)
        X_train = TensorDataset(X, y)

        fit_loop = range(self.n_epochs)
        if self.verbose > 1:
            fit_loop = tqdm(fit_loop, unit='epochs')
        
        for _ in fit_loop:
            for X_batch, y_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):
                
                y_pred   = self.model(X_batch)
                mse_loss = self.mse_loss(y_pred, y_batch)
                kl_loss  = self.kl_loss(self.model)
                loss     = mse_loss + self.kl_weight * kl_loss

                if self.save_loss:
                    self.loss_hist.append(loss)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
    
    def predict(self, X):

        f   = [self.model(Tensor(X)).detach().numpy() for _ in range(self.n_preds)]
        f   = np.concatenate(f, axis=-1)
        std = np.std(f, axis=-1)
        f   = np.mean(f, axis=-1)

        return f, std
