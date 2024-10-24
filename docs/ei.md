## Description

To target potentials for large improvements more precisely, the EGO algorithm aims to maximise expected improvement $EI(\theta) = \mathbb{E}[I(\theta)]$. In predictive posterior agnostic settings EI is calculated using the MC approximation.

$$
EI(\theta) \approx \frac{1}{M} \sum_{j=1}^M \max \{ 0, f_{min} - x_j \},
$$

where $\{x_j\}_{j=1}^M, x_j \sim \nu(\theta)$. As $M \rightarrow \infty$ the approximation becomes exact. If $\nu(\theta)$ is Gaussian with mean $\mu(\theta)$ and variance $\sigma^2(\theta)$, e.g., as in Gaussian Processes, then EI accepts the following closed form.

$$
EI(\theta) = \sigma(\theta) \left[ z(\theta) \cdot \Phi\left( z(\theta) \right) + \phi\left( z(\theta) \right)\right],
$$

where $z(\theta) = \frac{f_{min} - \mu(\theta)}{\sigma(\theta)}$ and $\Phi$ and $\phi$ are Gaussian CDF and PDF respectively.

## Usage

```python
from waggon.acquisitions import EI
```

