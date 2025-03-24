# bayesic-ode-estimator

```bayesic-ode-estimator``` is a general-purpose Python package for Bayesian parameter estimation in ordinary differential equation (ODE) models. The package uses MCMC sampling via the Metropolis–Hastings algorithm to estimate model parameters from noisy data, making it a robust tool for users working with dynamical systems.

## Features

- **General ODE Support:**  
  Easily apply the framework to a wide range of ODE models.
  
- **Customizable Priors:**  
  Define truncated-normal or other types of priors to encode your prior knowledge and enforce parameter constraints.
  
- **Likelihood Modeling:**  
  Incorporate Gaussian likelihoods to account for measurement noise in observed data.
  
- **Efficient MCMC Sampling:**  
  Utilize the Metropolis–Hastings algorithm to efficiently explore complex posterior distributions.
  
- **Extensible and Modular:**  
  The design allows for easy integration with custom models and likelihood functions.

## Installation

You can install **bayesic-ode-estimator** via pip:

```bash
pip install bayesic-ode-estimator
```

## Documentation

For detailed usage instructions and examples, plrease refer to sample Jupyter Notebook available on this GitHub repository.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on the GitHub repository.

## License

This project is licesned under the MIT License. See the LICENSE file for details.