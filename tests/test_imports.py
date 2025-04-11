import pytest
import sys


MODULES = [
    "waggon",
    "waggon.functions",
    "waggon.functions.landscapes",
    "waggon.functions.utils",
    "waggon.acquisitions",
    "waggon.optim",
    "waggon.optim.baselines",
    "waggon.surrogates",
    "waggon.utils"
]

@pytest.mark.parametrize("module", MODULES)
def test_import(module: str):
    __import__(module)
