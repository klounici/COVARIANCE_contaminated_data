#!/usr/bin/env python
"""
main auxiliary functions for missing values experiments

Authors: Karim Lounici and Grégoire Pacreau
"""

import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.covariance import MinCovDet
from statsmodels.robust.scale import huber
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.covariance import MinCovDet
from time import time

##### Low rank matrix generation #####

def low_rank(effective_rank, p, bell_curve=False):
    """
    Generates a low rank matrix using one of two spectral profile

    Parameters
    ----------
    effective_rank : TYPE
        Effective rank of the matrix.
    p : TYPE
        Dimension of the square matrix to be produced.
    bell_curve : TYPE, optional
        Whether to substitute the exponentially declining eigenvalues with
        a bell curve like eigenvalue structure. The default is False.

    Returns
    -------
    a tuple composed of the generated covariance matrix and the tuple of its eigenvalue decomposition.

    """

    ## Index of the singular values 
    singular_ind = numpy.arange(p, dtype=numpy.float64)

    if bell_curve:
        # Build the singular profile by assembling signal and noise components (from the make_low_rank_matrix code of numpy)
        tail_strength = 0.5
        low_rank = (1 - tail_strength) * numpy.exp(-1.0 * (singular_ind / effective_rank) ** 2)
        tail = tail_strength * numpy.exp(-0.1 * singular_ind / effective_rank)
        eigenvalues = (low_rank + tail)
    
    else:
        # Build the singular profile using an exponential:
        eigenvalues = 1/effective_rank*numpy.exp(-singular_ind/(effective_rank))

    assert np.abs(np.sum(eigenvalues)/np.max(eigenvalues) - effective_rank) < 1, 'effective rank not respected' 

    diag= numpy.eye(p)*eigenvalues
    
    # Generating random orthonormal vectors
    H = numpy.random.randn(p, p)
    u, s, vh = numpy.linalg.svd(H)
    mat = u @ vh

    sigma = mat @ diag @ mat.T

    # ensures variances are around 1
    s_diag = np.diag(sigma)
    sigma = sigma / np.max(s_diag)

    return sigma, (mat, eigenvalues/np.max(s_diag))

def estimate_cov(X, delta, mask, bias=True):
    """
    Unbiased covariance estimation under missing from Lounici (2014)

    Parameters
    ----------
    X : (n,p) numpy array or similar
        The samples from which to estimate the covariance. The covariance matrix will have shape (p,p).
    delta : float
        probability of the bernoulli random variable governing whether we see the values or not.
    mask : (n,p) boolean array or similar
        The boolean mask hiding values from the estimator, optional if X already has missing values.
    bias : bool, optional
        Whether to use the unbiased version of the classical estimator, should not have much impact. The default is True.

    Returns
    -------
    sigma_tilde : (p,p) numpy array
        Estimated covariance matrix.

    """
    if delta==1.:
        return numpy.cov(X.T, bias=bias)
    if delta==0:
        # if delta=0, we don't observe anything
        return numpy.zeros(X.shape)
    Y = mask * X
    sigma = numpy.cov(Y.T, bias=bias)
    sigma_tilde = (1/delta - 1/(delta**2)) * numpy.eye(sigma.shape[0]) * sigma.diagonal() + 1/(delta**2) * sigma
    return sigma_tilde

def MV_estimator(X, mask=None, bias=True, mcar=True, missing=0):
    """
    More general version of the unbiased covariance estimation under missing from Lounici (2014)

    Parameters
    ----------
    X : (n,p) numpy array or similar
        The samples from which to estimate the covariance. The covariance matrix will have shape (p,p).
    mask : (n,p) boolean array or similar or None, optional
        The boolean mask hiding values from the estimator, optional if X already has missing values.
    bias : bool, optional
        Whether to use the unbiased version of the classical estimator, should not have much impact. The default is True.
    mcar: bool, optional
        Whether or not the data should be treated as just Missing at Random and not Missing Completely at Random. If MAR is True, a delta will be estimated for each dimension independently. If False, a global delta will be used.
    missing : int or string, optional
        Value of the cell indicating a missing value. The first step of the procedure is to compute the empirical covariance matrix on the data, so all cells containing this value will be cast to 0.
        
    Returns
    -------
    sigma_tilde : (p,p) numpy array
        Estimated covariance matrix.

    """    
    X_new = X.copy()
    if missing != 0 and missing is not None:                # all missing values are set to 0
        X_new[X_new == missing] = 0

    if mask is None:                                        # if no specific values highlighted, drop all cells with value 0
        mask = (X_new != 0).astype(int)
    else:
        X_new = mask*X_new

    sigma = np.cov(X_new.T, bias=bias)                      # computing the biased covariance matrix

    if mcar:                                                # if MCAR, compute delta using all the missing values
        deltas = np.ones(X_new.shape[1]) * mask.sum()/mask.shape[0]/mask.shape[1]
    else:                                                   # otherwise compute the deltas column by column
        deltas = np.sum(mask, axis=0)/mask.shape[0]
    assert 0 not in deltas, 'A dimension has no values'

    correction = 1/np.outer(deltas, deltas)
    correction_diag = np.eye(deltas.shape[0]) * 1/deltas
    correction = correction - correction.diagonal + correction_diag
    sigma_tilde = correction * sigma
    return sigma_tilde

##### Adversarial perturbations #####

def contaminate_bernoulli(X, epsilon, intensity=1, option="gauss"):
    """
    bernoulli contamination of a gaussian random variable: random elements are
    replaced by a uniform law in [-1,1] or a normal law
    """
    assert option in ["gauss", "uniform", "dirac"], "Error in contamination procedure: {} not recognised, must be one of gauss, uniform or dirac".format(option)
    bernoulli_mask = numpy.random.binomial(p=epsilon, n=1, size=X.shape)
    if option == "gauss":
        noise = numpy.random.normal(0, intensity, size=X.shape)
    elif option == "uniform":
        noise = intensity*numpy.random.uniform(-1, 1, size=X.shape)
    elif option == "dirac":
        noise = intensity*numpy.ones(X.shape)
    X_contaminated = X*(1-bernoulli_mask) + noise*bernoulli_mask
    return X_contaminated, bernoulli_mask

def random_sparse_sphere(p, s):
    # generates s sparse vectors on the unit sphere
    sparsity_padding = numpy.array([0]*(p-s))
    coord = numpy.random.normal(0, 1, size=s)
    coord = numpy.concatenate([sparsity_padding, coord])
    coord = numpy.random.permutation(coord)
    norm = numpy.linalg.norm(coord)
    return coord/norm

def find_sparse_orthogonal(u, s=5, max_iter=1000):
    # Finds the sparsest unitary vector orthogonal to u

    # actually approximate experiment, where we generate s sparse vectors and select the one of lowest scalar product with u
    
    min_inner_product = 10
    best_vector = None

    for k in range(int(max_iter)):
        v = random_sparse_sphere(u.shape[0], s)
        inner_product = numpy.abs(v@u)
        if best_vector is None or inner_product < min_inner_product:
            min_inner_product = inner_product
            best_vector = v
        if inner_product == 0:
            break
    return best_vector

def contaminate_adversarial(X, sigma, epsilon=0.05, max_iter=1e3):
    """
    Adversarial contamination of X based on sparse vectors orthogonal to the first eigenspace of sigma.

    Parameters
    ----------
    X : (n,p) array of similar
        Samples of a mutlivariate distribution with covariance sigma.
    sigma : (p,p) array or similar
        Covariance matrix of X.
    s : sparsity parameter of the adversarial vector, ie number of non zero components (default is 5)
    max_iter : number of iterations in the search for the most orthogonal sparse vector.

    Returns
    -------
    X_noisy : (n,p) numpy array
        Contaminated samples of X.

    """
    u,d,v = numpy.linalg.svd(sigma)
    s = int(len(v[0]) * epsilon)
    if s == 0:
        return X
    theta_adv = find_sparse_orthogonal(v[0], s=s, max_iter=1e5)
    sigma_norm = numpy.linalg.norm(sigma)

    contaminated_row_mask = numpy.random.binomial(n=1, p=epsilon, size=X.shape[0])
    gaussian_noise = numpy.random.normal(0, 1, X.shape)
    contamination = numpy.outer(contaminated_row_mask, theta_adv)

    X_noisy = X + numpy.sqrt(2)*sigma_norm*gaussian_noise*contamination
    return X_noisy

##### Robust estimators #####

def MV_estimator(X, delta=0.9, N=1):
    """
    Estimator based on the random hiding of values followed by the unbiased estimator of Lounici (2014)

    Parameters
    ----------
    X : (n,p) array or similar.
        The contaminated data.
    delta : float between 0 and 1, optional
        Probability parameter of the bernoulli law for the random mask. The default is 0.9.
    N : int, optional
        Number of random bernoulli masks to be used during the estiamtion. The final matrix will be averaged
        over all the masks. The default is 1.

    Returns
    -------
    A (p,p) estimated covariance matrix.

    """
    mask = numpy.random.binomial(size=X.shape, n=1, p=delta)
    estimated_covs = estimate_cov(X, delta, mask)
    for k in range(N-1):
        mask = numpy.random.binomial(size=X.shape, n=1, p=delta)
        estimated_covs += estimate_cov(X, delta, mask)
    return estimated_covs/N

def remove_perturbation(X, threshold_ratio):
    # under assumption that X follows a multivariate normal distribution
    # here, mask is whether to keep data!
    try:
        stds = huber(X)[1]
    except:
        stds = np.std(X)
    stds_good_shape = numpy.outer(numpy.ones(X.shape[0]), stds)
    outliers = (X >= (threshold_ratio * stds_good_shape)) + (X <= (-threshold_ratio * stds_good_shape))
    mask_hat = 1-outliers
    mask_hat = numpy.multiply(mask_hat, 1) # bool -> int
    delta_hat = numpy.count_nonzero(mask_hat)/mask_hat.shape[0]/mask_hat.shape[1]
    return(mask_hat, delta_hat)

def MV_thresholding_estimator(X, threshold_ratio):
    mask_thresh, delta_thresh = remove_perturbation(X, threshold_ratio)
    return estimate_cov(X, delta_thresh, mask_thresh)

##### Comparison experiments #####

def get_proj(sigma, index):
    v = numpy.linalg.svd(sigma)[2][index]
    return numpy.outer(v, v.T)

##### Estimator handle #####

from source.cellHandler import *

def mask_union(A,B):
    return A + B - A*B

def apply_estimator(method, X, mask=None, delta=0.7, q=0.99):
    # here mask, if not none, should be ones where outliers and 0 if not
    X[numpy.isnan(X)] = 0
    missingMask = numpy.isnan(X).astype(int)
    tstart = time()
    if method == "DI":
        _, sigma = DI(X)
    elif method == "MinCovDet":
        est = MinCovDet().fit(X)
        sigma = est.covariance_
    elif method == "TSGS":
        _, sigma, _ = TSGS(X)
    elif method == "DDCMV95":
        # mask must be wether to keep the value, hence is 1 - isOutlier
        isOutlier = DDC(X, 0.95).astype(int)
        isOutlier = mask_union(isOutlier, missingMask)
        mask = 1 - isOutlier
        delta = mask.sum()/mask.shape[0]/mask.shape[1]
        sigma = estimate_cov(X, delta=delta, mask=mask)
    elif method == "DDCMV90":
        # mask must be wether to keep the value, hence is 1 - isOutlier
        isOutlier = DDC(X, 0.90).astype(int)
        isOutlier = mask_union(isOutlier, missingMask)
        mask = 1 - isOutlier
        delta = mask.sum()/mask.shape[0]/mask.shape[1]
        sigma = estimate_cov(X, delta=delta, mask=mask)
    elif method == 'DDCMV':
        isOutlier = DDC(X, q)
        isOutlier = mask_union(isOutlier, missingMask)
        mask = 1 - isOutlier
        delta = mask.sum()/mask.shape[0]/mask.shape[1]
        sigma = estimate_cov(X, delta=delta, mask=mask)
    elif method == "randomMV":
        sigma = MV_estimator(X, delta=delta)
    elif method == "tailMV":
        sigma = MV_thresholding_estimator(X, threshold_ratio=3)
    elif method == "classical":
        sigma = np.cov(X.T)
    elif method == "contMV":
        sigma = estimate_cov_cont(X, delta)
    elif method == "oracleMV":
        delta = (1-mask).sum()/mask.shape[0]/mask.shape[1]
        sigma = estimate_cov(X, delta, 1-mask)
    elif method == 'DDCKNN':
        mask, delta = remove_outliers_DDC(X, quantile=q)
        missing_data = X*mask
        imputer = KNNImputer(n_neighbors=5, weights="uniform", missing_values=0)
        knn_data = imputer.fit_transform(missing_data)
        sigma = np.cov(knn_data.T)
    elif method == 'DDCII':
        mask, delta = remove_outliers_DDC(X, quantile=q)
        missing_data = X*mask
        imputer = IterativeImputer(max_iter=10, random_state=0, missing_values=0)
        MI_data = imputer.fit_transform(missing_data)
        sigma = np.cov(MI_data.T)
    else:
        print("Method {} is unknown".format(method))
    tend = time()
    return sigma, tend-tstart

def remove_outliers_DDC(X, quantile=0.99):
    # beware: here mask has value 1 for outliers
    isOutlier = DDC(X, quantile)
    isOutlier = isOutlier.astype(float)
    isOutlier = numpy.nan_to_num(isOutlier,nan=0)
    isOutlier = isOutlier.astype(int)

    mask = 1 - isOutlier
    delta = mask.sum()/mask.shape[0]/mask.shape[1]
    return mask, delta

def compute_error(A, B, index=None, ord=2):
    # Computes operator error between A and B.
    # If index is not none but is int, computes the operator error between the projectors of
    # the [index]th eigenspace's projector
    if index is None:
        return numpy.linalg.norm(A-B, ord=ord)
    else:
        _, _, sv = numpy.linalg.svd(A)
        _, _, sv_hat = numpy.linalg.svd(B)
        theta = numpy.outer(sv[index], sv[index].T)
        theta_hat = numpy.outer(sv_hat[index], sv_hat[index].T)
        return numpy.linalg.norm(theta - theta_hat, ord=ord)