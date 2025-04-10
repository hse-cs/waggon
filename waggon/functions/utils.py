import torch
import random
import contextlib
import numpy as np


@contextlib.contextmanager
def fixed_random_seed(seed: int):
    """
    Context manager to set specified seed for random package

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@contextlib.contextmanager
def fixed_numpy_seed(seed: int):
    """
    Context manager to set specified seed for NumPy

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def fixed_torch_seed(seed: int):
    """
    Context manager to set specified for Torch
    """
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.cuda.set_rng_state_all(cuda_state)


def set_all_seed(seed: int):
    """
    Set specified seed for random, numpy, torch packages

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def fixed_all_seed(seed: int):
    """
    Context manager to set specified seed for random, numpy and torch packages

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all()

    set_all_seed(seed)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(torch_cuda_state)
