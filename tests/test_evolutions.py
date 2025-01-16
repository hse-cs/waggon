import argparse

import waggon.functions as f
from waggon.optim import DifferentialEvolutionOptimizer

FUNCS = {
    'ackley':     f.ackley,
    'himmelblau': f.himmelblau,
    'holder':     f.holder,
    'levi':       f.levi,
    'rosenbrock': f.rosenbrock,
    'tang':       f.tang,
    'thc':        f.three_hump_camel
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--function', help='Function to optimize', default='levi', choices=['thc', 'ackley', 'levi', 'himmelblau', 'rosenbrock', 'tang', 'holder']
    )
    parser.add_argument(
        '-d', '--dimensions', type=int, help='Dimensionality of the experiment', default=None
    )
    parser.add_argument(
        '-v', '--verbose', type=int, help='increase output verbose', choices=[0, 1, 2], default=1
    )
    parser.add_argument(
        '-e', '--experiments', help="Number of experiments", default=5, type=int
    )
    parser.add_argument(
        '-eps', '--epsilon', help="Required error rate", default=1e-1, type=float
    )
    parser.add_argument(
        '-n', '--n_candidates', help="Number of points in generation", type=int, default=11 ** 2
    )

    args = parser.parse_args()

    for i in range(args.experiments):
        print(f"Experiment #{i}")

        func = FUNCS[args.function]
        dim = args.dimensions

        if dim is not None:
            func = func(dim)
        else:
            func = func()
        
        eps = args.epsilon
        n_candidates = args.n_candidates
        
        evolution = DifferentialEvolutionOptimizer(
            func=func, eps=eps, n_candidates=n_candidates
        )
        evolution.optimise()


if __name__ == "__main__":
    main()
