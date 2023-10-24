#!/usr/bin/env python
"""
Main script

Authors: Karim Lounici and Grégoire Pacreau
"""

import argparse
import numpy
from source.run_experiments import run_experiment

parser=argparse.ArgumentParser(
    description="main experiment framework"
)
parser.add_argument(
    '--dim_size',
    type=int,
    default=100
)
parser.add_argument(
    '--sample_size',
    type=int,
    default=1000
)
parser.add_argument(
    '--effective_rank',
    type=int,
    default=5
)
parser.add_argument(
    '--method',
    type=str,
    choices=["DI",
             "TSGS",
             "randomMV",
             "DDCMV90",
             "DDCMV95",
             "tailMV",
             "classical",
             "contMV",
             "oracleMV",
             "all",
             "all+"],
    default="TSGS"
)
parser.add_argument(
    '--contamination',
    type=str,
    choices=["adversarial",
             "bernoulli"],
    default="bernoulli"
)
parser.add_argument(
    '--min_epsilon',
    type=float,
    default=0.01
)
parser.add_argument(
    '--max_epsilon',
    type=float,
    default=0.20
)
parser.add_argument(
    '--n_epsilon',
    type=int,
    default=7
)
parser.add_argument(
    '--n_exp',
    type=int,
    default=100
)
parser.add_argument(
    '--output',
    type=str,
    default="results_clean/"
)

parser.add_argument(
    '--intensity',
    type=float,
    default=1
)

parser.add_argument(
    '--contoption',
    type=str,
    choices=["gauss",
             "uniform",
             "dirac",
             "None"],
    default=None
)

parser.add_argument(
    '--time',
    type=bool,
    default=False
)

args = parser.parse_args()

if __name__ == '__main__':
    if args.method == "all":
        methods = ["DDCMV90", "DDCMV95", "tailMV", "classical", "oracleMV"]
    elif args.method == "all+":
        methods = ["MinCovDet", "DI", "TSGS", "DDCMV90", "tailMV", "classical", "oracleMV"]
    else:
        methods = [args.method]

    epsilons = numpy.linspace(args.min_epsilon, args.max_epsilon, args.n_epsilon)

    print("\n ########################################################################## \n")

    for m in methods:
        try :
            run_experiment(args.sample_size,
                         args.dim_size,
                         args.effective_rank,
                         args.n_exp,
                         epsilons,
                         args.output,
                         args.contamination,
                         m,
                         intensity=args.intensity,
                         cont_option=args.contoption)
        except:
            print("method {} failed".format(m))