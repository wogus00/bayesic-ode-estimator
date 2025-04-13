"""
lotka_volterra_example.py

Demonstration of using bayesic_ode_estimator to sample from the posterior
of a 4-parameter Lotka-Volterra model with fixed initial conditions.
"""

import numpy as np
import time

from bayesic_ode_estimator.ode_models import simulate_lotka_volterra
from bayesic_ode_estimator.likelihood import log_posterior_lv
from bayesic_ode_estimator.samplers import metropolis_hastings_single_chain

def main():
    # 1) "True" parameters & data generation
    alpha_true, beta_true, delta_true, gamma_true = 0.8, 0.04, 0.02, 0.5
    x0_given, y0_given = 30.0, 4.0  # known

    t_eval = np.linspace(0, 10, 100)
    sol_true = simulate_lotka_volterra(
        params=[alpha_true, beta_true, delta_true, gamma_true],
        t_span=(t_eval[0], t_eval[-1]),
        y0=[x0_given, y0_given],
        t_eval=t_eval
    )
    lv_clean = sol_true.y.T  # (100, 2)

    # Add noise
    np.random.seed(123)
    data_observed = lv_clean + np.random.normal(scale=5.0, size=lv_clean.shape)

    # 2) Define priors for alpha, beta, delta, gamma
    prior_means = [1.0, 0.05, 0.03, 0.6]
    prior_sds   = [0.1, 0.01, 0.01, 0.1]
    sigma_noise = 10.0

    # 3) Log-posterior callback
    def lv_log_posterior(params):
        return log_posterior_lv(
            params      = params,
            t_eval      = t_eval,
            data        = data_observed,
            x0          = x0_given,
            y0          = y0_given,
            prior_means = prior_means,
            prior_sds   = prior_sds,
            sigma       = sigma_noise
        )

    # 4) Run multiple chains
    n_iter = 3000
    n_chains = 4
    chains = []
    acc_rates = []

    start_time = time.time()
    for chain_id in range(n_chains):
        # Simple initial guess (jitter around prior means)
        init_guess = np.array(prior_means) * (1.0 + 0.2 * np.random.randn(len(prior_means)))
        init_guess = np.maximum(init_guess, 1e-3)

        chain, acc = metropolis_hastings_single_chain(
            init_params      = init_guess,
            log_posterior_fn = lv_log_posterior,
            n_iter           = n_iter,
            proposal_scales  = [0.1 * sd for sd in prior_sds],
            random_seed      = 1000 + chain_id,
            chain_id         = chain_id,
            verbose          = True
        )
        chains.append(chain)
        acc_rates.append(acc)

    end_time = time.time()
    print(f"\nAll chains completed in {end_time - start_time:.2f} seconds.")

    # 5) Summaries
    burn_in = n_iter // 2
    combined = np.concatenate([c[burn_in:] for c in chains], axis=0)
    post_mean = np.mean(combined, axis=0)
    post_std  = np.std(combined, axis=0)

    for i, (chain, acc) in enumerate(zip(chains, acc_rates)):
        print(f"Chain {i}: final sample={chain[-1]}, acceptance={acc:.3f}")

    print("\nPosterior mean:", post_mean)
    print("Posterior std:", post_std)
    print("True values: alpha={:.2f}, beta={:.3f}, delta={:.3f}, gamma={:.2f}".format(
        alpha_true, beta_true, delta_true, gamma_true
    ))


if __name__ == "__main__":
    main()
