## Description

Surrogate-based optimisation (SBO) carries out optimisation using a surrogate model. SBO does not strive for global accuracy. Instead it only requires the surrogate model to be sufficiently accurate to guide optimisation toward the true optimum. SBO can be particularly useful in several scenarious. For example, when the original model is computationally expensive, or when data is noise, or both [1].

`SurrogateOptimiser` is based on the `Optimiser` class. To run, it requires the experiment, [`waggon.Function`](https://hse-cs.github.io/waggon/functions/), the surrogate model, [`waggon.Surrogate`](https://hse-cs.github.io/waggon/custom_surr/), and the acquisition function, [`waggon.acquisition`](https://hse-cs.github.io/waggon/custom_acqf/).

## Usage

```python
import waggon
from waggon.acquisitions import WU
from waggon.surrogates.gan import WGAN_GP as GAN
from waggon.test_functions import three_hump_camel

from waggon.optim import SurrogateOptimiser

# initialise the function to be optimised
func = three_hump_camel()
# initialise the surrogate to carry out optimisation
surr = GAN()
# initialise optimisation acquisition function
acqf = WU()

# initialise optimiser
opt = SurrogateOptimiser(func=func, surr=surr, acqf=acqf)

# run optimisation
opt.optimise()

# visualise optimisation results
waggon.utils.display()
```

[1] Martins, J.R.R.A. and Ning, S.A. (2022) Surrogate-based Optimisation in *Engineering design optimization*. Cambridge, United Kingdom: Cambridge University Press. 