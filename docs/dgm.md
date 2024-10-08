## Description

Having a generator model that would map a simple distribution, $\mathcal{Z}$, to a complex and unknown distribution, $p(\mu)$, is desired in many settings as it allows for the generation of samples from the intractable data space. For our purposes, conditional deep generative models (DGMs) are used to model the stochastic response of the simulator. This allows to model essentially any response shape.

The goal of a conditional DGM is to obtain a generator $G: \mathbb{R}^s \times \Theta \rightarrow \mathbb{R}^d$ such that distributions $G(\mathcal{Z}; \theta)$ and $p(\mu; \theta)$ match for each $\theta \in \Theta$. Since $\mathcal{Z}$ and $p(\mu)$ are independent, we get $G(\mathcal{Z}; \theta) \sim p(\mu; \theta)$. For simplicity of notation we denote $G(\mathcal{Z}; \theta)$ as $G(\theta)$.

As per [2], the package contains the implementation of a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to be used with Wasserstein uncertainty.

<!-- Thus, we explore the search space by training a deep generative surrogate $G$ on the set of ground truths $\mathcal{M}$. For each value $\theta \in \Theta$, we can generate the corresponding response of the simulator $\nu (\theta) = G(\theta)$, i.e., $\nu$ is the predictive distribution at $\theta$. -->


## Usage

```python
from waggon.surrogates.gan import WGAN
```

[2] Tigran Ramazyan, Mikhail Hushchyn and Denis Derkach. "Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space." Arxiv abs/2407.1117 (2024). [[arxiv]](https://arxiv.org/abs/2407.11917)