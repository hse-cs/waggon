## Description

Bayesian optimisation is typically equipped with a Gaussian process (GP), which is defined by a mean function and a kernel function that describes the shape of the covariance. BO with a GP surrogate requires covariance matrix inversion with $O(n^3)$ cost in terms of the number of observations, which is challenging for a large number of responses and in higher dimensions. To make BO scalable [6, 7, 8, 9] consider a low-dimensional lin- ear subspace and decompose it into subsets of dimensions, i.e., some structural assumptions are required that might not hold. In addition, GP requires a proper choice of kernel is crucial. BO may need the construction of new kernels [10]. [11] proposes a greedy approach for automatically building a kernel by combining basic kernels such as linear, exponential, periodic, etc., through kernel summation and multiplication.

## Usage

```python
from waggon.surrogates import GP
```

[6] N.deFreitasandZ.Wang.Bayesian optimization in high dimensions via random embeddings. In Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence, 2013.
‌
[7] J. Djolonga, A. Krause, and V. Cevher. High-dimensional gaussian process bandits. Advances in neural information processing systems, 26, 2013.

[8] R. Garnett, M. A. Osborne, and P. Hennig. Active learning of linear embeddings for gaussian processes. arXiv preprint arXiv:1310.6740, 2013.

[9] M. Zhang, H. Li, and S. Su. High dimensional bayesian optimization via supervised dimension reduction. arXiv preprint arXiv:1907.08953, 2019.

[10] N. S. Gorbach, A. A. Bian, B. Fischer, S. Bauer, and J. M. Buhmann. Model selection for gaussian process regression. In Pattern Recognition: 39th German Conference, GCPR 2017, Basel, Switzerland, September 12–15, 2017, Proceedings 39, pages 306–318. Springer, 2017.

[11] D. Duvenaud. Automatic model construction with Gaussian processes. PhD thesis, University of Cambridge, 2014.