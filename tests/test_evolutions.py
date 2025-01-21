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
        '-f', '--function', help='Function to optimize', default='levi', 
        choices=['thc', 'ackley', 'levi', 'himmelblau', 'rosenbrock', 'tang', 'holder']
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
    parser.add_argument(
        '-err', '--error_type', help="Method of error computation", choices=['x', 'f'], default='x'
    )
    parser.add_argument(
        '-iter', '--max_iter', help='Number of maximum iterations', type=int, default=1000
    )
    parser.add_argument(
        '-seed', '--lhs_seed', help='Seed', type=int, default=None
    )
    parser.add_argument(
        '-no_save', '--save_results', help='Flag to save results', action="store_false"
    )
    parser.add_argument(
        '-p', '--path', help='Directory where experiment results will be saved', type=str, default='test_results'
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
        
        error_type = args.error_type
        eps = args.epsilon
        n_candidates = args.n_candidates
        max_iter = args.max_iter
        lhs_seed = args.lhs_seed

        save_results = args.save_results
        path = args.path
        
        evolution = DifferentialEvolutionOptimizer(
            func=func, 
            eps=eps, 
            n_candidates=n_candidates, 
            error_type=error_type,
            max_iter=max_iter,
            lhs_seed=lhs_seed,
            save_results=save_results,
            path=path,
        )
        evolution.optimise()


if __name__ == "__main__":
    main()
