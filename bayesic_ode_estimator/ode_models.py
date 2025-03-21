"""
ode_models.py

Defines ODE functions and solver helpers for Bayesian ODE estimation.
"""

import numpy as np
from scipy.integrate import solve_ivp

def lotka_volterra_ode(t, y, alpha, beta, delta, gamma):
    """
    Lotka–Volterra system of ODEs:
      dx/dt = alpha * x - beta * x * y
      dy/dt = delta * x * y - gamma * y
    
    Parameters
    ----------
    t : float
        Time.
    y : array_like, shape (2,)
        [x, y] = [prey, predator] at time t.
    alpha, beta, delta, gamma : float
        ODE parameters.
    
    Returns
    -------
    dydt : list of floats
        [dx/dt, dy/dt].
    """
    x, y_pop = y
    dx_dt = alpha * x - beta * x * y_pop
    dy_dt = delta * x * y_pop - gamma * y_pop
    return [dx_dt, dy_dt]


def simulate_lotka_volterra(params, t_span, y0, t_eval=None):
    """
    Solve the Lotka–Volterra system using scipy.integrate.solve_ivp.

    Parameters
    ----------
    params : array_like, shape (4,)
        [alpha, beta, delta, gamma].
    t_span : tuple(float, float)
        (t_start, t_end).
    y0 : array_like, shape (2,)
        [x0, y0] initial conditions.
    t_eval : array_like, optional
        Times at which to save the ODE solution.

    Returns
    -------
    sol : OdeResult
        The result from solve_ivp with sol.y and possibly sol.t.
    """
    alpha, beta, delta, gamma = params
    sol = solve_ivp(
        fun=lambda t, state: lotka_volterra_ode(t, state, alpha, beta, delta, gamma),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        dense_output=True
    )
    return sol
