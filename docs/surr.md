## Description

Surrogate-based modelling is an established technique for design optimisation. A surrogate, or meta, model is introduced to approximate the black-box function. The surrogate model is, typically, updated at each iteration of the optimisation loop.

## Usage

Our package considers two types of surrogate models - `Surrogate` and `GenSurrogate`. The essential difference being that `Surrogate` is used in a discriminative setting only, while `GenSurrogate` has the ability to sample from the predicted posterior distribution, i.e., it is used in generative setting. Both classes can be inherited and used to make a custom surrogate model that would fit the `SurrogateOptimiser` pipeline.

```python
from waggon.surrogates import Surrogate, GenSurrogate
```


