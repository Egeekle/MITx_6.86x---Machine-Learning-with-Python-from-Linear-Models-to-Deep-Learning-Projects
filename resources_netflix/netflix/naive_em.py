"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    log_likelihood = 0.0
    
    for i in range(n):
        # Compute log probability for each component
        log_probs = np.zeros(K)
        for j in range(K):
            # Log of Gaussian PDF: log P(x|μ, σ²)
            # For spherical Gaussian: log(1/(2πσ²)^(d/2)) - 0.5 * ||x-μ||² / σ²
            diff = X[i, :] - mixture.mu[j, :]
            squared_dist = np.sum(diff ** 2)
            # Add small epsilon to variance to avoid numerical issues
            var_j = max(mixture.var[j], 1e-10)
            log_prob = -0.5 * d * np.log(2 * np.pi * var_j) - 0.5 * squared_dist / var_j
            # Add log of prior P(z=j)
            # Use small epsilon to avoid log(0) if p[j] is exactly 0
            log_probs[j] = log_prob + np.log(max(mixture.p[j], 1e-300))
        
        # Use logsumexp for numerical stability
        log_sum = logsumexp(log_probs)
        log_likelihood += log_sum
        
        # Compute posterior probabilities: P(z=j|x) = exp(log_probs[j] - log_sum)
        for j in range(K):
            post[i, j] = np.exp(log_probs[j] - log_sum)
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    
    # Compute weighted counts
    n_hat = post.sum(axis=0)
    
    # Update mixture weights
    p = n_hat / n
    
    # Update means
    mu = np.zeros((K, d))
    for j in range(K):
        if n_hat[j] > 0:
            mu[j, :] = (post[:, j].reshape(-1, 1) * X).sum(axis=0) / n_hat[j]
    
    # Update variances
    var = np.zeros(K)
    for j in range(K):
        if n_hat[j] > 0:
            # Compute weighted squared distances
            diff = X - mu[j, :]
            squared_dists = np.sum(diff ** 2, axis=1)
            var[j] = (post[:, j] * squared_dists).sum() / (d * n_hat[j])
            # Ensure variance is at least a small epsilon
            var[j] = max(var[j], 1e-10)
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = None
    log_likelihood = None
    
    while prev_log_likelihood is None or log_likelihood - prev_log_likelihood > 1e-6:
        prev_log_likelihood = log_likelihood
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
    
    return mixture, post, log_likelihood
