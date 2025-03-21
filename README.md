from setuptools import setup, find_packages

setup(
    name="bayesic-ode-estimator",
    version="0.1.0",
    description="A generalized Python package for PDE estimation using Bayesian methods.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/wogus00/bayesic-pde-estimator",
    packages=find_packages(),  # This will find pde, sampler, and utils subpackages
    install_requires=[
        # Add package dependencies here, e.g. "numpy", "scipy", etc.
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
