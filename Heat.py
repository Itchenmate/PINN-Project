#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
@author: Chen Gadi & Aviv Burshtein
"""

"""
We'll solve the heat equation on the domain  x∈[0,1] and  t∈[0,1]
using the finite difference method. We'll use Dirichlet boundary conditions and a
simple initial condition for this example.
"""
# %%
import numpy as np
import tensorflow as tf
from pyDOE import lhs
import sys
import pandas as pd
import os

# Settings of the directory
sys.path.insert(0, 'Utils/')
from Utils.plotting import plot_coefficients_distribution,plot_coefficients_histogram, plot_initial_condition, plot_results_3d_heat, plot_error_heatmap_heat, plot_loss_history

from PDE_HEAT import HeatPINN

import numpy as np

# %%
def simulate_heat_equation(x, t, coeffs=None, f=None, alpha=0.01, dt=0.001):
    """
    Simulate the heat equation using the finite difference method.

    Parameters:
    - x: array of spatial points in [0,1]
    - t: array of time points in [0,1]
    - coeffs: coefficients of the polynomial (used if f is None)
    - f: initial condition function
    - alpha: thermal diffusivity
    - dt: time step for simulation

    Returns:
    - u: array of temperatures at points x and time t
    """

    # Ensure x and t are numpy arrays
    x = np.array(x).flatten()
    t = np.array(t).flatten()

    # Define spatial domain and time interval for simulation
    x_domain = np.linspace(0, 1, 100)
    dx = x_domain[1] - x_domain[0]

    u_vals = []
    for xi, ti in zip(x, t):
        times = np.arange(0, ti + dt, dt)

        # Initialize u based on given conditions
        if f:
            u = f(x_domain)
        else:
            u = np.sum([coeffs[i] * (x_domain ** i) for i in range(6)], axis=0)

        # Time-stepping using finite differences
        for _ in times:
            u_xx = np.roll(u, -1) - 2 * u + np.roll(u, 1) / dx ** 2
            u = u + alpha * dt * u_xx

        # Interpolate to get the value at the desired spatial point xi
        u_val = np.interp(xi, x_domain, u)
        u_vals.append(u_val)

    return np.array(u_vals)


# %%
if __name__ == "__main__":
    np.random.seed(5890)
    tf.random.set_seed(5890)

    lb = 0.0
    ub = 1.0

    N0 = 50
    N_b = 50
    N_f = 2000

    x0 = np.random.uniform(0, 1, size=(N0, 1))
    coeffs = np.random.uniform(-1, 1, size=(6, 1))
    u0 = simulate_heat_equation(x0, 0, coeffs)

    tb = np.random.uniform(0, 1, size=(N_b, 1))
    x_lb = np.array([0.0] * N_b).reshape(-1, 1)
    x_ub = np.array([1.0] * N_b).reshape(-1, 1)

    X_f = lb + (ub - lb) * lhs(2, N_f)
    x_vals = np.linspace(0, 1, 100)
    t_vals = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_vals, t_vals)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    layers = [8, 20, 20, 20, 1]

    # Convert to tensors
    x0 = tf.expand_dims(tf.convert_to_tensor(x0[:, 0]), axis=-1)
    x_lb = tf.expand_dims(tf.convert_to_tensor(x_lb[:, 0]), axis=-1)
    x_ub = tf.expand_dims(tf.convert_to_tensor(x_ub[:, 0]), axis=-1)
    tb = tf.expand_dims(tf.convert_to_tensor(tb[:, 0]), axis=-1)
    X_f = tf.convert_to_tensor(X_f)
    X_star = tf.convert_to_tensor(X_star)
    if len(u0.shape) > 1:
        u0 = tf.expand_dims(tf.convert_to_tensor(u0[:, 0]), axis=-1)
    else:
        u0 = tf.expand_dims(tf.convert_to_tensor(u0), axis=-1)

    # %%
        ########################################
        #                                      #
        #          DATA PREPARATION            #
        #                                      #
        ########################################
        """
     DDD     A   TTTTT   A         PPPP  RRRR  EEEEE PPPP    A   RRRR    A   TTTTT IIIII  OOO  N   N 
    D  D   A A    T    A A        P   P R   R E     P   P  A A  R   R  A A    T     I   O   O NN  N 
    D   D AAAAA   T   AAAAA       PPPP  RRRR  EEEEE PPPP  AAAAA RRRR  AAAAA   T     I   O   O N N N 
    D  D  A   A   T   A   A       P     R R   E     P     A   A R R   A   A   T     I   O   O N  NN 
    DDD   A   A   T   A   A       P     R  RR EEEEE P     A   A R  RR A   A   T   IIIII  OOO  N   N 
        """

    model = HeatPINN(x0, u0, coeffs, x_ub, x_lb, tb, X_f[:, 0:1], X_f[:, 1:2], X_star, ub, lb, layers)

    ##### Set the number of iterations ( 0 for off)
    adam_iterations = 1000  # Number of training steps
    lbfgs_max_iterations = 2000  # Max iterations for lbfgs


    ##### Training
    Adam_hist, LBFGS_hist, elpases = model.train(adam_iterations, lbfgs_max_iterations)

##### final prediction
# Reshape X_star to have 10,000 rows and 2 columns
    X_star_reshaped = np.array([X_star[:, 0].numpy().flatten(), X_star[:, 1].numpy().flatten()]).T

# Predict for all grid points
    u_pred = model.predict(X_star_reshaped[:, 0], X_star_reshaped[:, 1])
    u_pred = np.array(u_pred).reshape(-1, 1)  # Ensure u_pred is a column vector

#### simulate:
    u_true = np.array([simulate_heat_equation([xi], [ti], coeffs) for xi, ti in zip(X_star[:, 0], X_star[:, 1])])

##### final error
    u_pred_reshaped = u_pred.reshape(100, 100)
    u_true_reshaped = u_true.reshape(100, 100)
    error_u = np.linalg.norm(u_true_reshaped - u_pred_reshaped, 2) / np.linalg.norm(u_true_reshaped, 2)

    print('Error u: %e' % (error_u))

# %%
    x_vals = np.linspace(0, 1, 100)
    t_vals = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x_vals, t_vals)
    print("Shape of X:", X.shape)
    print("Shape of T:", T.shape)
    print("Shape of u_pred_reshaped:", u_pred_reshaped.shape)
    u_pred_reshaped = u_pred.reshape(X.shape)

    # Ensure the directory 'figures' exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Plotting (res ,pred and loss )
    fig_res = plot_results_3d_heat(u_pred_reshaped, u_true_reshaped, X, T)
    fig_err = plot_error_heatmap_heat(u_pred_reshaped, u_true_reshaped, X, T)
    fig_los_his = plot_loss_history(Adam_hist, LBFGS_hist)
    fig1 = plot_initial_condition(x0.numpy().flatten(), u0.numpy().flatten())
    fig2 = plot_coefficients_distribution(coeffs)
    fig3 = plot_coefficients_histogram(coeffs)


    # Save the figures
    fig_res.savefig('figures/fig_res_3d.png')
    fig_err.savefig('figures/fig_err_3d.png')
    fig_los_his.savefig('figures/fig_loss_history.png')
    fig1.savefig('figures/initial_condition_plot.png')
    fig2.savefig('figures/coefficients_distribution.png')
    fig3.savefig('figures/coefficients_histogram.png')





