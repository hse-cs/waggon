from .base import Surrogate

import torch
import numpy as np
import torch.nn as nn
from torchensemble import AdversarialTrainingRegressor
from torch.utils.data import TensorDataset, DataLoader

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else


class Regressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, hidden_size=16):
        super(Regressor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_outputs)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class DE(Surrogate):
    def __init__(self, **kwargs):
        super(DE, self).__init__()

        self.name         = 'DE'
        self.model        = kwargs['model'] if 'model' in kwargs else None
        self.n_estimators = kwargs['n_estimators'] if 'n_estimators' in kwargs else 10
        self.hidden_size  = kwargs['hidden_size'] if 'hidden_size' in kwargs else 64
        self.n_epochs     = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.lr           = kwargs['G_lr'] if 'G_lr' in kwargs else 1e-3
        self.batch_size   = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        self.verbose      = kwargs['verbose'] if 'verbose' in kwargs else 1
        self.opt          = kwargs['opt'] if 'opt' in kwargs else 'Adam'
        self.weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        self.device       = kwargs['device'] if 'device' in kwargs else torch.device('cpu')

        # if 'scheduler' in kwargs:
        #     self.model.set_scheduler('StepLR',
        #                             step_size = kwargs['sched_step'] if 'sched_step' in kwargs else self.n_epochs//3,
        #                             gamma = kwargs['sched_gamma'] if 'sched_gamma' in kwargs else 0.1)
    
    def fit(self, X, y):

        if self.model == None:
            base_estimator = Regressor(n_inputs=X.shape[-1], n_outputs=y.shape[-1], hidden_size=self.hidden_size)
            self.model = AdversarialTrainingRegressor(estimator = base_estimator, n_estimators = self.n_estimators)
            self.model.device = self.device
        
        self.model.set_optimizer(self.opt, lr = self.lr, weight_decay = self.weight_decay)
        
        X = Tensor(X)
        y = Tensor(y)
        
        X -= torch.min(X)
        X /= torch.max(X)
        
        X_train = TensorDataset(X, y)
        
        train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.model.fit(train_loader, epochs=self.n_epochs, log_interval=1000, save_model=False)
    
    def predict(self, X):

        X = Tensor(X)
        X -= torch.min(X)
        X /= torch.max(X)
        
        f   = [estimator(X).detach().numpy() for estimator in self.model.estimators_]
        f   = np.concatenate(f, axis=-1)
        std = np.std(f, axis=-1)
        f   = np.mean(f, axis=-1)

        return f, std
