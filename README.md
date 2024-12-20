# Welcome to WAGGON: WAssrestein Global Gradient-free OptimisatioN

[![PyPI version](https://badge.fury.io/py/waggon.svg)](https://badge.fury.io/py/waggon.svg)
[![Documentation](https://img.shields.io/badge/documentation-yes-green.svg)](https://hse-cs.github.io/waggon)
[![Downloads](https://static.pepy.tech/badge/waggon)](https://pepy.tech/project/waggon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`WAGGON` is a python library of black box gradient-free optimisation. Currently, the library contains implementations of optimisation methods based on Wasserstein uncertainty and baseline approaches from the following papers:

- Tigran Ramazyan, Mikhail Hushchyn and Denis Derkach. Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space, 2024.[[arxiv]](https://arxiv.org/abs/2407.11917) [[ECAI 2024 Proceedings]](https://ebooks.iospress.nl/doi/10.3233/FAIA240765)

<!-- ![](https://github.com/hse-cs/waggon/blob/master/images/readme_image.png) -->

## Implemented methods
- Wasserstein Uncertainty Global Optimisation (WU-GO)
- Bayesian optimisation: via Expected Improvement (EI), Lower and Upper Confidence Bounds (LCB, UCB)

## Installation

```
pip install waggon
```
or
```
git clone https://github.com/hse-cs/waggon
cd waggon
pip install -e
```

## Basic usage

(See more examples in the [documentation](https://hse-cs.github.io/waggon/).)

The following code snippet is an example of surrogate optimisation.

```python
import waggon
from waggon.optim import SurrogateOptimiser

from waggon.acquisitions import WU
from waggon.surrogates.gan import WGAN_GP as GAN
from waggon.test_functions import three_hump_camel

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

# visualise
waggon.utils.display()
```


## Support

- Home: [https://github.com/hse-cs/waggon](https://github.com/hse-cs/waggon)
- Documentation: [https://hse-cs.github.io/waggon](https://hse-cs.github.io/waggon)
- For any usage questions, suggestions and bugs please use the [issue page](https://github.com/hse-cs/waggon/issues).

<!-- ## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/probaforms/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/probaforms" />
</a> -->
