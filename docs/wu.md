## Description

Ref.[2] suggests a reformulation of the lower confidence bound (LCB) acquisition function, by using Wasserstein balls to quantify uncertainty. Thus moving uncertainty estimation to the Wasserstein space, and accounting for the shape of black-box response.

For a predictive posterior $\nu$ its uncertainty can be formulated as

$$
F(\nu) = \sup_{\mu \in \bar{B}_{\varepsilon}(\nu)} \mathbb{E}^{\mu} [f],
$$

where $\bar{B}_{\varepsilon}(\nu)$ is a Wasserstein ball centered at $\nu$ with radius $\varepsilon$.

For each candidate $\nu$ we take the radius of its Wasserstein ball as the distance to the closest ground truth $\left( \mathcal{M} - set of all ground truths \right)$.

$$
\varepsilon = \inf_{\mu \in \mathcal{M}} \mathbb{W}_2(\nu, \mu).
$$

Under minor assumptions the supremum is achieved at the boundary. This result is used to reformulate regret for a predictive posterior given by a generative surrogate as $G(\theta)$ as follows:

$$
\mathcal{R}_{\mathbb{W}}(\theta) = \mu(\theta) - \kappa \cdot \inf_{\mu \in \mathcal{M}} \mathbb{W}_2(G(\theta), \mu).
$$

## Usage

```python
from waggon.acquisitions import WU
```

[2] Tigran Ramazyan, Mikhail Hushchyn and Denis Derkach. "Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space." Arxiv abs/2407.1117 (2024). [[arxiv]](https://arxiv.org/abs/2407.11917)