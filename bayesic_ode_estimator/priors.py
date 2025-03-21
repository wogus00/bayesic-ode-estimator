"""
priors.py

Implements truncated Normal priors and other possible prior distributions.
"""

import numpy as np
import math
from math import inf
from scipy.special import erf

def truncated_normal_logpdf(x, mu, sd, lower=0.0, upper=np.inf):
    """
    Log PDF of a Normal(mu, sd) truncated to [lower, upper].
    Returns -∞ if x is outside.
    """
    if not (lower < x < upper):
        return -inf
    
    z = (x - mu) / sd
    z_lower = (lower - mu) / sd
    z_upper = (upper - mu) / sd
    
    pdf_z = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * z**2)
    
    def stdnorm_cdf(val):
        return 0.5 * (1.0 + erf(val / np.sqrt(2)))
    
    cdf_lower = stdnorm_cdf(z_lower) if z_lower > -inf else 0.0
    cdf_upper = stdnorm_cdf(z_upper) if z_upper < inf else 1.0
    
    Z = cdf_upper - cdf_lower
    if Z < 1e-15:
        return -inf
    
    return math.log(pdf_z) - math.log(sd) - math.log(Z)


def log_prior_truncnormal(params, prior_means, prior_sds):
    """
    Sums the log of truncated Normal(>0) for a list of parameters.

    Returns
    -------
    lp : float
        The sum of log-priors. -∞ if invalid.
    """
    lp = 0.0
    for val, mu, sd in zip(params, prior_means, prior_sds):
        if val <= 0:
            return -inf
        lp_comp = truncated_normal_logpdf(val, mu, sd, 0.0, np.inf)
        if np.isinf(lp_comp):
            return -inf
        lp += lp_comp
    return lp
