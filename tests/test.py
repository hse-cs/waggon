import argparse
from waggon.utils import display
from waggon import functions as f
from waggon.optim import SurrogateOptimiser
from waggon.acquisitions import WU, WU_IDW, EI, CB
from waggon.surrogates import BNN, DE, DGP, GAN, GP

FUNCS = {
    'ackley':     f.ackley,
    'himmelblau': f.himmelblau,
    'holder':     f.holder,
    'levi':       f.levi,
    'rosenbrock': f.rosenbrock,
    'tang':       f.tang,
    'thc':        f.three_hump_camel
}

SURR = {
    'bnn' : BNN(),
    'de' :  DE(),
    'dgp' : DGP(),
    'gan' : GAN(),
    'gp' :  GP()
}

ACQF = {
    'wu':  WU(),
    'wu-idw': WU_IDW(),
    'ei':  EI(),
    'lcb': CB(minimise=True),
    'ucb': CB(minimise=False)
}

SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 73]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', help='optimise the given function', default='thc', choices=['thc', 'ackley', 'levi', 'himmelblau', 'rosenbrock', 'tang', 'holder'])
    parser.add_argument('-d', '--dimensions', type=int, help='dimensionality of the experiment', default=None)
    parser.add_argument('-s', '--surrogate', help='surrogate for optimisation', default='gan', choices=['gan', 'bnn', 'de', 'dgp', 'gp'])
    parser.add_argument('-a', '--acquisition', help='acqusition function to use for optimisation', default='wu', choices=['wu', 'wu-idw', 'ei', 'lcb', 'ucb'])
    parser.add_argument('-v', '--verbose', type=int, help='increase output verbose', choices=[0, 1, 2], default=1)

    args = parser.parse_args()
    
    if (args.surrogate == 'gan' and not ('wu' in args.acquisition)) or (args.surrogate != 'gan' and 'wu' in args.acquisition):
        raise ValueError(f'Surrogate {args.surrogate} is not compatible with {args.acquisition} acquisition function')

    for i, seed in enumerate(SEEDS):
        
        print(f'Experiment #{i}')

        opt = SurrogateOptimiser(func=FUNCS[args.function](args.dimensions) if args.dimensions else FUNCS[args.function](),
                                surr=SURR[args.surrogate],
                                acqf=ACQF[args.acquisition],
                                verbose=args.verbose,
                                seed=seed,
                                save_results=True)
        opt.optimise()
        display(opt)

if __name__ == '__main__':
    main()
