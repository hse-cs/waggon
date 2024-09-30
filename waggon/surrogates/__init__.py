from .de import DE
from .gp import GP
from .bnn import BNN
from .dgp import DGP
from .gan import WGAN_GP as GAN
from .base import Surrogate, GenSurrogate

__all__ = [
    'BNN',
    'DE',
    'DGP',
    'GAN',
    'GP',
    'Surrogate',
    'GenSurrogate',
]