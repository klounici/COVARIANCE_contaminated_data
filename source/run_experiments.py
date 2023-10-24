#!/usr/bin/env python
"""
Main function for experiments

Authors: Karim Lounici and Grégoire Pacreau
"""

from source.auxiliary_functions import *
import numpy
from tqdm import tqdm
import pickle

def run_experiment(sample_size,
                    dim_size,
                    effective_rank,
                    n_exp, epsilons,
                    output,
                    contamination,
                    method,
                    cont_option=None,
                    intensity=None, 
                    seed=0,
                    index=None,
                    return_time=True):

    """
    Runs one experiment on one set of parameters.

    Parameters
    ----------
    sample_size : int
        Number of samples to be generated from the distribution.
    dim_size : int
        Dimension of the distribution.
    effective_rank : float
        Approximation of the required effective rank of the covariance matrix. Be aware that the actual effecrive rank may vary from this parameter to this parameter plus one.
    n_exp : int
        Number of repetitions of the experiment.
    epsilons : array of float
        List of the contamination parameters to be explored during this experiment.
    output: string
        Name of the output folder where the results will be stored.
    contamination: string, can be Bernoulli or Adversarial
        Contamination type. Note that the Adversarial was created with Huber contamination in mind. It is as yet unknown if it is a good candidate for an adversarial contamation in the cell-wise setting.
    method: string or list of strings
        Outlier correction methods to be tested during this experiment. If a list, each method will be saved in a seperate file.
    cont_option: string, can be Gauss, Dirac or Uniform, optional
        Distribution of the contaminations.
    intenisity : float, optional
        Intensity of the contamination as defined in the paper. If set to 0, allows for missing values experiments.
    seed : int, optional
        Random seed of the experiment.
    return_time: bool, optional
        Whether to compute and return execution times for each methods.

    Returns
    ----------
    """

    filename = 'nsample={}_dim={}_erank={}_nexp={}_cont={}_meth={}_intensity={}_cont-type={}.pkl'.format(
        sample_size,
        dim_size,
        effective_rank,
        n_exp,
        contamination,
        method,
        intensity,
        cont_option
    )

    if not os.path.exists(output+filename):

        numpy.random.seed(seed)
        estimates = {}
        exec_times = {}

        estimates["truth"] = {}

        assert isinstance(method, str) or isinstance(method, list), "methods should be a string or a list of string, got {}".format(type(method))
        if isinstance(method, str):
            method_name = method
            estimates[method] = {}
            exec_times[method] = {}
        else:
            method_name = '+'.join(method)
            for k in method:
                estimates[k] = {}
                exec_times[k] = {}

        methods = list(estimates.keys())
        methods.remove("truth")

        pbar = tqdm(epsilons)
        pbar.set_description(method + ' ' + contamination)
        for epsilon in pbar:

            # initialising the experiments in dictionary
            for k in estimates.keys():
                estimates[k][epsilon] = []
            for k in methods:
                exec_times[k][epsilon] = []

            for _ in range(n_exp):
                #generate data and saving the true covariance matrix
                sigma,_ = low_rank(effective_rank, dim_size)
                estimates["truth"][epsilon].append(sigma)

                X = numpy.random.multivariate_normal(numpy.zeros(dim_size), sigma, size=sample_size)
            
                # contaminate data
                if contamination == "bernoulli":
                    X_noisy, mask = contaminate_bernoulli(X, epsilon, intensity, option=cont_option)
                elif contamination == "adversarial":
                    X_noisy = contaminate_adversarial(X, sigma, epsilon)
                    mask = np.zeros(X.shape)
                #compute and save covariance estimates
                for k in methods:
                    try:
                        sigma_hat, exec_time = apply_estimator(k, X_noisy, mask)
                    except:
                        sigma_hat, exec_time = np.ones(sigma.shape)*np.nan, np.nan
                    estimates[k][epsilon].append(sigma_hat)
                    exec_times[k][epsilon].append(exec_time)

        with open(output+filename, 'wb+') as file:
            pickle.dump(estimates, file)
        if return_time:
            with open(output+'exectime_'+filename, 'wb+') as file:
                pickle.dump(exec_times, file)
    else:
        print("{} already exists".format(filename))