## Description

Deep Gaussian processes (DGP) [5] attempt to resolve the issue of finding the best kernel for a GP. That is done by stacking GPs in a hierarchical struc- ture as Perceptrons in a multilayer perceptron, but the number of variational parameters to be learnt by DGPs increases linearly with the number of data points, which is infeasible for stochastic black-box optimisation, and they have the same matrix inverting issue as GPs, which limits their scalability.

## Usage

```python
from waggon.surrogates import DGP
```

[5‌] Damianou, A. and Lawrence, N. (2013). Deep Gaussian Processes. PMLR. [online] Available at: https://proceedings.mlr.press/v31/damianou13a.pdf
‌
