# Black-box Optimisation

Simulation of real-world experiments is key to scientific discoveries and engineering solutions. Such techniques use parameters describing configurations and architectures of the system. A common challenge is to find the optimal configuration for some objective, e.g., such that it maximises efficiency or minimises costs and overhead.

The simulator is treated as a black box experiment. This means that observations come from an unknown and likely difficult-to-estimate function. Using a surrogate model for black-box optimisation (BBO) is an established technique, as BBO has a rich history of both gradient and gradient-free methods, most of which come from tackling problems that arise in physics, chemistry and engineering.

The optimisation problem can be defined as:

$$
\inf_{\theta \in \Theta} f(\theta),
$$

where $\Theta$ is the search space. If the black-box function is stochastic, the problem accepts the following form:

$$
\inf_{\theta \in \Theta} \mathbb{E}\left[ f(\theta, x) \right].
$$

And in case of distributionally robust optimistion, the objective becomes:

$$
\inf_{\theta \in \Theta} \sup_{\mu \in \mathcal{P}} \mathbb{E}_{x \sim \mu} \left[ f(\theta, x) \right],
$$

where $\mathcal{P}$ is an ambiguity set, i.e., set of viable distributions.