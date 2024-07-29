import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else 


class Generator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=64):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x_cond, x_noise):
        x = torch.cat((x_cond, x_noise), dim=1)
        x = self.model(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, n_inputs, hidden_size=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    
class WGAN_GP(object):# TODO: add cuda
    def __init__(self, **kwargs):
        self.name        = 'GAN'
        self.G           = kwargs['G'] if 'G' in kwargs else None
        self.D           = kwargs['D'] if 'D' in kwargs else None
        self.batch_size  = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 64
        self.n_epochs    = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.n_disc      = kwargs['n_disc'] if 'n_disc' in kwargs else 5
        self.latent_dim  = kwargs['latent_dim'] if 'latent_dim' in kwargs else 10
        self.scaling     = kwargs['scaling'] if 'scaling' in kwargs else False
        self.lambda_gp   = kwargs['lambda_gp'] if 'lambda_gp' in kwargs else 1
        self.G_lr        = kwargs['G_lr'] if 'G_lr' in kwargs else 1e-4
        self.D_lr        = kwargs['D_lr'] if 'D_lr' in kwargs else 1e-4
        self.scheduler   = kwargs['scheduler'] if 'scheduler' in kwargs else False
        self.verbosity   = kwargs['verbosity'] if 'verbosity' in kwargs else 1
        self.save_loss   = kwargs['save_loss'] if 'save_loss' in kwargs else False

        if self.save_loss:
            self.G_loss_hist = []
            self.D_loss_hist = []
    

    def gradient_pen(self, cond_data, real_test_data, gen_data):

        alpha = Tensor(np.random.random((real_test_data.size(0), 1)))

        inter = (alpha * real_test_data + ((1 - alpha) * gen_data)).requires_grad_(True)
        D_inter = self.D(torch.cat((cond_data, inter), dim=1))
        fake = Variable(Tensor(real_test_data.size(0), 1).fill_(1.0))

        gradients = torch.autograd.grad(
            outputs      = D_inter,
            inputs       = inter,
            grad_outputs = fake,
            create_graph = True,
            retain_graph = True,
            only_inputs  = True,
        )[0]
        
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp
        
        
    def fit(self, X, y, **kwargs):

        self.verbosity = kwargs['verbosity'] if 'verbosity' in kwargs else self.verbosity

        if self.G is None:
            self.G = Generator(n_inputs=X.shape[-1] + self.latent_dim, n_outputs=y.shape[-1], hidden_size=self.hidden_size)
        if self.D is None:
            self.D = Discriminator(n_inputs=X.shape[-1] + y.shape[-1], hidden_size=self.hidden_size)
        
        X = Tensor(X)
        y = Tensor(y)
        X_train = TensorDataset(X, y)
        
        if self.scheduler:
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.G_lr*1e2)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.D_lr*1e2)
            
            G_scheduler = StepLR(self.G_opt, step_size=self.n_epochs//3, gamma=0.1)
            D_scheduler = StepLR(self.D_opt, step_size=self.n_epochs//3, gamma=0.1)
        else:
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.G_lr)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.D_lr)
        
        self.G.train(True)
        self.D.train(True)
        
        if self.verbosity > 1:
            fit_loop = tqdm(range(self.n_epochs), unit="epoch")
        else:
            fit_loop = range(self.n_epochs)
        for e in fit_loop:
            for X_batch, y_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):

                for _ in range(self.n_disc):
                    noise_batch = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))

                    y_gen_batch = self.G(X_batch, noise_batch)

                    D_real = self.D(torch.cat((X_batch, y_batch), dim=1))
                    D_fake = self.D(torch.cat((X_batch, y_gen_batch.detach()), dim=1))
                    
                    D_loss = torch.mean(D_fake) - torch.mean(D_real)
                    grad_p = self.gradient_pen(X_batch, y_batch, y_gen_batch.detach())
                    D_loss += self.lambda_gp * grad_p

                    self.D_opt.zero_grad()
                    D_loss.backward()
                    self.D_opt.step()
                    
                    if self.save_loss:
                        self.D_loss_hist.append(D_loss.detach().numpy())
                
                noise_batch = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))

                y_gen_batch = self.G(X_batch, noise_batch)
                
                D_fake = self.D(torch.cat((X_batch, y_gen_batch), dim=1))
                
                G_loss = -torch.mean(D_fake)
            
                self.G_opt.zero_grad()
                G_loss.backward()
                self.G_opt.step()
                
                if self.save_loss:
                    self.G_loss_hist.append(G_loss.detach().numpy())
            
            if self.verbosity > 1:
                fit_loop.set_description(f"G loss: {G_loss:.4f}, D loss: {D_loss:.4f}")
            
            if self.scheduler:
                G_scheduler.step()
                D_scheduler.step()
            
        self.G.train(False)
        self.D.train(False)
    
    
    def sample(self, X_cond):
        noise = Variable(Tensor(np.random.normal(0, 1, (X_cond.shape[0], self.latent_dim))))
        X_gen = self.G(Tensor(X_cond).detach(), noise.detach())
        return X_gen.detach().numpy()
    
    
    def _scale(self, X):
        
        ss = StandardScaler()
        ss.fit(X)
        X_ss = ss.transform(X)
        
        return X_ss
    
    
    def predict(self, X, scheduler=False):
        
        if self.scaling:
            X = self._scale(X)
        
        X, X_cond = X[:, :1], X[:, 1:]

        if X_cond.shape[-1] == 1:
            X_cond = X_cond.reshape(-1, 1)

        self.scheduler = scheduler
        
        self.fit(X, X_cond)
