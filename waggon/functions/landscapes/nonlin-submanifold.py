import numpy as np
from ..base import FunctionV2


class NonlinearSubmanifold(FunctionV2):
    def __init__(self, dim=20, subdim=8, **kwargs):
        super().__init__(**kwargs)

        raise NotImplementedError(
            "TO BE DONE SOON!"
        )