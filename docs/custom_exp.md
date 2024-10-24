## Description

Any experiment or test function can be added to the proposed surrogate optimisation pipeline. For the experiment to be compatible with the rest of the package, one ought to inherit the `waggon.Function` class to create a new type of experiment.

## Usage

```python
from waggon.functions import Function

class my_Experiment(Function):
    ...
```