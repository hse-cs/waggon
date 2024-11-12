## Description

Bayesian inference allows us to learn a probability distribution over possible neural networks. By making a simple adjustment to standard neural network methods, we can approximately solve this inference problem. The resulting approach reduces overfitting, facilitates learning from small datasets, and provides insights into the uncertainty of our predictions.

In Bayesian inference, instead of relying on a single point estimate of the weights, $w^*$, and its associated prediction function, $\hat{y}_{w^*}(x)$, as in conventional training, we infer distributions $p(w|D)$ and $p(\hat{y}_{w^*}(x)|D)$. One key advantage of using distributions for model parameters and predictions is the ability to quantify uncertainty, such as by calculating the variance [3].

## Usage

```python
from waggon.surrogates import BNN
```


[3] Xu, W., Ricky, Li, X. and Duvenaud, D. (2022). Infinitely Deep Bayesian Neural Networks with Stochastic Differential Equations. PMLR, pp.721–738. Available [online](https://proceedings.mlr.press/v151/xu22a).
‌