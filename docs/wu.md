## Description

Bayesian optimisation is typically equipped with a Gaussian process (GP), which is defined by a mean function and a kernel function that describes the shape of the covariance. BO with a GP surrogate requires covariance matrix inversion with $O(n^3)$ cost in terms of the number of observations, which is challenging for a large number of responses and in higher dimensions. To make BO scalable [6, 7, 8, 9] consider a low-dimensional lin- ear subspace and decompose it into subsets of dimensions, i.e., some structural assumptions are required that might not hold. In addition, GP requires a proper choice of kernel is crucial. BO may need the construction of new kernels [10]. [11] proposes a greedy approach for automatically building a kernel by combining basic kernels such as linear, exponential, periodic, etc., through kernel summation and multiplication.

## Usage

```python
from waggon.acquisitions import WU
```

[12]