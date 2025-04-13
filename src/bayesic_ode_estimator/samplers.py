"""
samplers.py

Implements MCMC sampling algorithms (e.g., Metropolis-Hastings).
"""

import numpy as np
from math import inf, log
from tqdm import tqdm
from matplotlib import pyplot as plt

def metropolis_hastings_single_chain(
    init_params,
    log_posterior_fn,
    n_iter=5000,
    proposal_scales=None,
    random_seed=None,
    chain_id=0,
    verbose=True
):
    """
    Generic Metropolis-Hastings sampler for a user-supplied log_posterior_fn.

    Parameters
    ----------
    init_params : array_like
        Initial guess for the parameters.
    log_posterior_fn : callable
        Function(params) -> float (the log-posterior).
    n_iter : int
        Number of MCMC iterations.
    proposal_scales : array_like or None
        Proposal std dev for each parameter. If None, defaults to 0.01 * init_params.
    random_seed : int or None
        If provided, sets the random seed.
    chain_id : int
        For logging.
    verbose : bool
        If True, prints acceptance rate at the end.

    Returns
    -------
    chain : ndarray, shape (n_iter, n_params)
        The MCMC samples at each iteration.
    acceptance_rate : float
        Fraction of proposals that were accepted.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    params = np.array(init_params, dtype=float)
    dim = len(params)
    chain = np.zeros((n_iter, dim), dtype=float)
    chain[0] = params
    
    # Evaluate log-posterior at the initial state
    current_lp = log_posterior_fn(params)
    
    if proposal_scales is None:
        # example default
        proposal_scales = 0.01 * np.abs(params)
        proposal_scales = np.maximum(proposal_scales, 1e-4)
    
    accepted = 0
    for i in tqdm(range(1, n_iter)):
        current_params = chain[i - 1]
        proposal = current_params + np.random.normal(scale=proposal_scales, size=dim)
        
        proposal_lp = log_posterior_fn(proposal)
        log_accept_ratio = proposal_lp - current_lp
        
        if log_accept_ratio > 0 or log(np.random.rand()) < log_accept_ratio:
            chain[i] = proposal
            current_lp = proposal_lp
            accepted += 1
        else:
            chain[i] = current_params
    
    acc_rate = accepted / (n_iter - 1)
    if verbose:
        print(f"[Chain {chain_id}] Acceptance rate: {acc_rate:.3f}")
    
    return chain, acc_rate

class MetropolisHastings:
    def __init__(self, log_posterior_fn, proposal_scales=None, random_seed=None):
        self.log_posterior_fn = log_posterior_fn
        self.proposal_scales = proposal_scales
        self.random_seed = random_seed

    def sample(self, init_params, n_iter=5000, n_chains=2, verbose=True):
        chains = []
        acc_rates = []
        for i in range(n_chains):
            print(f"Running chain {i}...")
            chain, acc = metropolis_hastings_single_chain(
                init_params=init_params,
                log_posterior_fn=self.log_posterior_fn,
                n_iter=n_iter,
                proposal_scales=self.proposal_scales,
                random_seed=self.random_seed,
                chain_id=i,
                verbose=verbose
            )
            chains.append(chain)
            acc_rates.append(acc)
        
        result = MCMCResult(chains, acc_rates)
        return result
    
class MCMCResult:
    def __init__(self, chains, acc_rates):
        self.chains = chains
        self.acc_rates = acc_rates
        self.burn_in = len(chains[0]) // 2
    
    def acceptance_rates(self, verbose=True):
        # Print acceptance rates if verbose
        if verbose:
            print("Acceptance rates:")
            for i, acc in enumerate(self.acc_rates):
                print(f"Chain {i}: {acc:.3f}")
            print("\n")
        return self.acc_rates
    
    def summary(self, burn_in=None):
        if burn_in is None:
            burn_in = len(self.chains[0]) // 2
        
        combined = np.concatenate([c[burn_in:] for c in self.chains], axis=0)
        post_mean = np.mean(combined, axis=0)
        post_std  = np.std(combined, axis=0)
        
        return post_mean, post_std
          
    def visualize(self, params, burn_in = None):
        if burn_in is None:
            burn_in = len(self.chains[0]) // 2
        combined = np.concatenate([c[burn_in:] for c in self.chains], axis=0)
        # plot the trace
        plt.figure(figsize=(12, 8))
        for i in range(combined.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.plot(combined[:, i])
            plt.title(f"Parameter {params[i]}")
        plt.tight_layout()
        plt.show()
        # plot the histogram
        plt.figure(figsize=(12, 8))
        for i in range(combined.shape[1]):
            plt.subplot(2, 2, i + 1)
            plt.hist(combined[:, i], bins=30)
            plt.title(f"Parameter {params[i]}")
        plt.tight_layout()
        plt.show()
