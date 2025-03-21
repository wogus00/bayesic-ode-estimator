"""
likelihood.py

Defines likelihood and posterior functions for ODE parameter estimation.
"""

import numpy as np
from math import inf, pi, log
from .ode_models import simulate_lotka_volterra  # relative import from same package
from .priors import log_prior_truncnormal

def log_likelihood_lv(params, t_eval, data, x0, y0, sigma=10.0):
    """
    Gaussian log-likelihood for Lotka-Volterra given observed data in shape (n_times, 2).
    """
    sol = simulate_lotka_volterra(params, (t_eval[0], t_eval[-1]), [x0, y0], t_eval=t_eval)
    if sol.y.shape[1] != len(t_eval):
        return -inf
    
    model_pred = sol.y.T
    if model_pred.shape != data.shape:
        return -inf
    
    diff = data - model_pred
    ssr = np.sum(diff**2)
    n_obs = data.size
    
    # -0.5 * sum((r/sigma)^2) - (n_obs/2)*log(2*pi*sigma^2)
    ll = -0.5 * (ssr / sigma**2) - 0.5 * n_obs * log(2 * pi * sigma**2)
    return ll


def log_posterior_lv(params, t_eval, data, x0, y0, prior_means, prior_sds, sigma=10.0):
    """
    log-posterior = log-prior + log-likelihood for the Lotka-Volterra example.
    """
    lp = log_prior_truncnormal(params, prior_means, prior_sds)
    if lp == -inf:
        return -inf
    ll = log_likelihood_lv(params, t_eval, data, x0, y0, sigma)
    return lp + ll
