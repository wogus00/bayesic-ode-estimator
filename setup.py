from setuptools import setup, find_packages

setup(
    name="bayesic_ode_estimator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    python_requires=">=3.7",
    author="Jaehyeon Park",
    description="A framework for Bayesian ODE estimation using MCMC and more.",
    url="https://github.com/wogus/bayesic-ode-estimator",  # or wherever
)
