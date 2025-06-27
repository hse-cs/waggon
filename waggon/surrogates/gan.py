from typing import Optional
from typing import Literal

import copy
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from .base import GenSurrogate


class Generator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=64):
        super().__init__()
        
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
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    
class WGAN_GP(GenSurrogate):
    def __init__(
        self,
        G: Optional[nn.Module] = None,
        D: Optional[nn.Module] = None,
        latent_dim: int = 10,
        hidden_size: int = 64,
        batch_size: int = 8,
        n_epochs: int = 100,
        n_disc: int = 5,
        G_lr: float = 1e-4,
        D_lr: float = 1e-4,
        lambda_gp: float = 1.0,
        scheduler: bool = False,
        verbose: Literal[0, 1, 2] = 1,
        save_loss: bool = False,
        device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    ):
        super().__init__()

        self.G = G
        self.D = D
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_disc = n_disc
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.lambda_gp = lambda_gp
        self.scheduler = scheduler
        self.verbose = verbose
        self.save_loss = save_loss

        self.G_loss_hist: list = []
        self.D_loss_hist: list = []
        
        self.D_scheduler = None
        self.G_scheduler = None

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
    

    def gradient_pen(self, cond_data, real_data, gen_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(
            batch_size, 1,
            dtype=torch.float32,
            device=self.device
        )
        inter_data = alpha * real_data + (1 - alpha) * gen_data
        inter_data.requires_grad = True

        disc_inter = self.D(torch.cat([cond_data, inter_data], dim=-1))
        fake = torch.ones(
            batch_size, 1,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        gradients, = torch.autograd.grad(
            outputs      = disc_inter,
            inputs       = inter_data,
            grad_outputs = fake,
            create_graph = True,
            retain_graph = True,
            only_inputs  = True,
        )
        
        gp = torch.mean((gradients.norm(2, dim=1) - 1).pow(2))
        return gp
        
        
    def fit(self, X, y, **kwargs):
        verbose = kwargs.pop("verbose", self.verbose)

        self.configure_models(X, y)
        self.configure_optimizers()

        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        train_dataset = data.TensorDataset(x_t, y_t)
        
        self.G.train(mode=True)
        self.D.train(mode=True)
        
        fit_loop = range(self.n_epochs)
        if self.verbose >= 2:
            fit_loop = tqdm.tqdm(fit_loop, unit="epoch", leave=True, position=2)
        
        for epoch in fit_loop:
            train_loader = data.DataLoader(
                train_dataset,
                batch_size = self.batch_size,
                shuffle=True,
            )

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.train_step(X_batch, y_batch)
            
            self.on_train_epoch_end(fit_loop)
        

    def configure_models(self, x: np.ndarray, y: np.ndarray):
        if self.G is None:
            self.G = Generator(
                n_inputs = x.shape[-1] + self.latent_dim,
                n_outputs = y.shape[-1],
                hidden_size = self.hidden_size
            ).to(self.device)
        
        if self.D is None:
            self.D = Discriminator(
                n_inputs = x.shape[-1] + y.shape[-1],
                hidden_size = self.hidden_size
            ).to(self.device)

    def configure_optimizers(self):
        if not self.scheduler:
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.G_lr)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.D_lr)
            return

        self.G_opt = torch.optim.Adam(
            self.G.parameters(), lr=self.G_lr * 100.0
        )
        self.D_opt = torch.optim.Adam(
            self.D.parameters(), lr=self.D_lr * 100.0
        )
            
        self.G_scheduler = lr_scheduler.StepLR(
            self.G_opt, step_size=self.n_epochs // 3, gamma=0.1
        )
        self.D_scheduler = lr_scheduler.StepLR(
            self.D_opt, step_size=self.n_epochs // 3, gamma=0.1
        )
    
    def on_train_epoch_end(self, fit_loop):
        if self.G_loss_hist and self.D_loss_hist and self.verbose >= 2:
            gen_loss = self.G_loss_hist[-1]
            disc_loss = self.D_loss_hist[-1]

            fit_loop.set_description(
                f"Gen. loss: {gen_loss:.3f}, Disc. loss: {disc_loss:.4f}"
            )
        
        if self.scheduler:
            self.G_scheduler.step()
            self.D_scheduler.step()
    
    def gen_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        batch_size = x_batch.size(0)

        noise_batch = torch.randn(
            batch_size, self.latent_dim,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        y_gen_batch = self.G(x_batch, noise_batch)
        disc_fake = self.D(torch.cat([x_batch, y_gen_batch], dim=1))
        gen_loss = -torch.mean(disc_fake)
            
        self.G_opt.zero_grad()
        gen_loss.backward()
        self.G_opt.step()
                
        if self.save_loss:
            self.G_loss_hist.append(gen_loss.item())
    
    def disc_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        batch_size = x_batch.size(0)

        noise_batch = torch.randn(
            batch_size, self.latent_dim,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        y_gen_batch = self.G(x_batch, noise_batch).detach()
        disc_real = self.D(torch.cat([x_batch, y_batch], dim=1))
        disc_fake = self.D(torch.cat([x_batch, y_gen_batch], dim=1))

        disc_loss = torch.mean(disc_fake) - torch.mean(disc_real)
        grad_pen = self.gradient_pen(x_batch, y_batch, y_gen_batch)
        disc_loss += self.lambda_gp * grad_pen

        self.D_opt.zero_grad()
        disc_loss.backward()
        self.D_opt.step()
                    
        if self.save_loss:
            self.D_loss_hist.append(disc_loss.item())

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        for _ in range(self.n_disc):
            self.disc_step(x_batch, y_batch)
        self.gen_step(x_batch, y_batch)

    def sample(self, X_cond: np.ndarray):
        if self.G is None:
            raise ValueError("Generator is not specified!")

        batch_size = X_cond.shape[0]
        noise = torch.randn(
            batch_size, self.latent_dim,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        batch_cond = torch.tensor(X_cond, dtype=torch.float32, device=self.device)
        X_gen = self.G(batch_cond, noise)
        return X_gen.cpu().detach().numpy()

    @property
    def history(self):
        return {
            "gen_loss": copy.deepcopy(self.G_loss_hist),
            "disc_loss": copy.deepcopy(self.D_loss_hist)
        }

    @property
    def name(self):
        return "WassersteinGAN"
