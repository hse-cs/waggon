import numpy as np
import pytest

from waggon.functions import Function
from waggon.functions import three_hump_camel
from waggon.functions import ackley
from waggon.functions import rosenbrock
from waggon.functions import submanifold_rosenbrock

from waggon.functions import FunctionV2
from waggon.functions import ThreeHumpCamel
from waggon.functions import Ackley
from waggon.functions import Rosenbrock
from waggon.functions import SubmanifoldRosenbrock
from waggon.functions import StyblinskyTang
from waggon.functions import Levi
from waggon.functions import Himmelblau
from waggon.functions import Holder

from utils import check_func_call_dims
from utils import check_func_sample_dims
from utils import check_func_log_transform


def _test_func(func, func_log):
    is_old_api = True
    if isinstance(func, FunctionV2):
        is_old_api = False
    
    check_func_call_dims(func, func_log, is_old_api)
    check_func_sample_dims(func, func_log, is_old_api)
    check_func_log_transform(func, func_log, is_old_api)


@pytest.mark.parametrize(
    "func, func_log", [
        # Old Three Hump Camel
        (three_hump_camel(log_transform=False), three_hump_camel(log_transform=True)),
        # New Three Hump Camel
        (ThreeHumpCamel(), ThreeHumpCamel(log_transform=True)),
    ]
)
def test_thc(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        # Old Ackley
        (ackley(dim=2, log_transform=False), ackley(dim=2)),
        (ackley(dim=3, log_transform=False), ackley(dim=3)),
        (ackley(dim=5, log_transform=False), ackley(dim=5)),
        (ackley(dim=10, log_transform=False), ackley(dim=10)),
        # New Ackley
        (Ackley(dim=2), Ackley(dim=2, log_transform=True)),
        (Ackley(dim=3), Ackley(dim=3, log_transform=True)),
        (Ackley(dim=5), Ackley(dim=5, log_transform=True)),
        (Ackley(dim=10), Ackley(dim=10, log_transform=True)),
    ]
)
def test_ackley(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        # Old API Rosenbrock
        (
            rosenbrock(dim=5, log_transform=False),
            rosenbrock(dim=5)
        ),
        (
            rosenbrock(dim=20, log_transform=False),
            rosenbrock(dim=20)
        ),
        # New API Rosenbrock
        (
            Rosenbrock(dim=7),
            Rosenbrock(dim=7, log_transform=True)
        ),
        (
            Rosenbrock(dim=20),
            Rosenbrock(dim=20, log_transform=True)
        )
    ]
)
def test_rosenbrock(func, func_log):
    _test_func(func, func_log)



@pytest.mark.parametrize(
    "func, func_log", [
        # Old Submanifold
        (
            submanifold_rosenbrock(dim=20, sub_dim=8, log_transform=False),
            submanifold_rosenbrock(dim=20, sub_dim=8)
        ), 
        (
            submanifold_rosenbrock(dim=6, sub_dim=1, log_transform=False),
            submanifold_rosenbrock(dim=6, sub_dim=1)
        ),
        (
            submanifold_rosenbrock(dim=12, sub_dim=12, log_transform=False),
            submanifold_rosenbrock(dim=12, sub_dim=12) 
        ),
        # New Submanifold
        (
            SubmanifoldRosenbrock(dim=20, sub_dim=8),
            SubmanifoldRosenbrock(dim=20, sub_dim=8, log_transform=True)
        ), 
        (
            SubmanifoldRosenbrock(dim=6, sub_dim=1),
            SubmanifoldRosenbrock(dim=6, sub_dim=1, log_transform=True)
        ),
        (
            SubmanifoldRosenbrock(dim=12, sub_dim=12),
            SubmanifoldRosenbrock(dim=12, sub_dim=12, log_transform=True) 
        )
    ]
)
def test_submanifold(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        (
            StyblinskyTang(dim=4),
            StyblinskyTang(dim=4, log_transform=True)
        ),
        (
            StyblinskyTang(dim=1),
            StyblinskyTang(dim=1, log_transform=True)
        ),
        (
            StyblinskyTang(dim=20),
            StyblinskyTang(dim=20, log_transform=True)
        )
    ]
)
def test_tang(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        (
            Levi(),
            Levi(log_transform=True)
        )
    ]
)
def test_levi(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        (
            Himmelblau(),
            Himmelblau(log_transform=True)
        )
    ]
)
def test_himmelblau(func, func_log):
    _test_func(func, func_log)


@pytest.mark.parametrize(
    "func, func_log", [
        (
            Holder(),
            Holder(log_transform=True)
        )
    ]
)
def test_holder(func, func_log):
    _test_func(func, func_log)
