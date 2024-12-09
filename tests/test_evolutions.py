from waggon import functions as f
from waggon.optim import SurrogateOptimiser
from waggon.optim import DifferentialEvolutionOptimizer
from waggon.functions import levi, rosenbrock

import numpy as np


def main():
    # TO DO: test optimzer on all test functions

    func = f.Function()
    e = DifferentialEvolutionOptimizer(func, save_results=False)

    print(e.predict())


if __name__ == "__main__":
    main()