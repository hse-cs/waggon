import time
import pytest

import random
import numpy as np
import torch

from waggon.utils import set_all_seed
from waggon.utils import fixed_random_seed
from waggon.utils import fixed_numpy_seed
from waggon.utils import fixed_torch_seed
from waggon.utils import fixed_all_seed


@pytest.mark.parametrize(
    "seed", range(20)
)
def test_set_all_seed(seed):
    set_all_seed(seed)
    # Generate for seed for first time
    x1_random = random.random()
    x1_numpy = np.random.randn(17, 23)
    x1_torch = torch.randn(13, 19)
    
    time_seed = round(time.time())
    set_all_seed(time_seed)
    set_all_seed(seed)
    
    # Generate for seed for second time
    x2_random = random.random()
    x2_numpy = np.random.randn(17, 23)
    x2_torch = torch.randn(13, 19)

    assert x1_random == x1_random
    assert np.all(x1_numpy == x2_numpy)
    assert torch.all(x1_torch == x2_torch)


@pytest.mark.parametrize(
    "seed", range(20)
)
def test_fixed_random_seed(seed):
    with fixed_random_seed(seed):
        x1 = random.random()
        y1 = random.randrange(100)
        z1 = random.uniform(0.0, 2.0)

    time_seed = round(time.time())
    set_all_seed(time_seed)

    with fixed_random_seed(seed):
        x2 = random.random()
        y2 = random.randrange(100)
        z2 = random.uniform(0.0, 2.0)

    assert x1 == x2
    assert y1 == y2
    assert z1 == z2 


@pytest.mark.parametrize(
    "seed", range(20)
)
def test_fixed_numpy_seed(seed):
    with fixed_numpy_seed(seed):
        x1 = np.random.normal(size=(25, 8))
        y1 = np.random.uniform(low=0.0, high=2.0, size=(25, 7))
        z1 = np.random.beta(1.5, 1.5, size=11)
    
    time_seed = round(time.time())
    set_all_seed(time_seed)

    with fixed_numpy_seed(seed):
        x2 = np.random.normal(size=(25, 8))
        y2 = np.random.uniform(low=0.0, high=2.0, size=(25, 7))
        z2 = np.random.beta(1.5, 1.5, size=11)

    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)


@pytest.mark.parametrize(
    "seed", range(20)
)
def test_fixed_torch_seed(seed):
    with fixed_torch_seed(seed):
        x1 = torch.randn(18, 17)
        y1 = torch.rand(2, 3, 5)
        
    time_seed = round(time.time())
    set_all_seed(time_seed)

    with fixed_torch_seed(seed):
        x2 = torch.randn(18, 17)
        y2 = torch.rand(2, 3, 5)

    assert torch.all(x1 == x2)
    assert torch.all(y1 == y2)


@pytest.mark.parametrize(
    "seed", range(30)
)
def test_fixed_all_seed(seed):
    with fixed_all_seed(seed):
        x_random = random.random()
        x_numpy = np.random.randn()
        x_torch = torch.rand(1).item()

    time_seed = round(time.time())
    set_all_seed(time_seed)

    with fixed_all_seed(seed):
        y_random = random.random()
        y_numpy = np.random.randn()
        y_torch = torch.rand(1).item()

    assert x_random == y_random
    assert x_numpy == y_numpy
    assert x_torch == y_torch

