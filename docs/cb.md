## Description

Lower confidence bound (LCB) is another common Bayesian optimisation approach. Its acquisition function is regret defined as follows:

$$
\mathcal{R}(\theta) = \mu(\theta) - \kappa \cdot \sigma(\theta).
$$

It is a linear combination of exploitation, $\mu(\theta)$, and exploration, $\sigma(\theta)$. The trade-off between the two is controlled via the hyperparameter $\kappa$. Smaller $\kappa$ yields more exploitation, and larger values of $\kappa$ yield more exploration of high-variance responses, where uncertainty is higher, i.e., where the black box is more unknown.

## Usage

```python
from waggon.acquisitions import LCB, UCB
```

