## Description

`WAGGON` focuses on optimisation methods, and `Optimiser` is a base class for optimisation algorithms. It contains common methods and properties, e.g., `optimise` that runs the optimisation loop until the chosen error, `error_type`, is small enough, `opt_eps`, and `create_candidates` that samples candidate points using Latin Hyperube sampling.

The class can be inherited for implementing specific approaches. Currently `waggon.optim` contains surrogate-based optimisation, `SurrogateOptimiser` (described in the following [section](https://hse-cs.github.io/waggon/surr_opt/)). New methods will be added as our research continues and can be suggested via a pull request.


<!-- ## Usage

```python

``` -->