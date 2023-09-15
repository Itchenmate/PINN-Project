#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""
import numpy as np
import tensorflow as tf
# Plotting
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata


# %%

######################################################################
############################# Plotting ###############################
######################################################################
def plot_results(u_pred, u_DataSet, x, t, x0, tb, lb, ub, x_f, t_f):
    plt.ioff()

    # Create the grid of the domain
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(X_star, u_DataSet[:, 0], (X, T), method='cubic')

    cmap_result = 'viridis'

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])  # Adjusted height_ratios

    # 3D plot for the entire domain
    ax0 = fig.add_subplot(gs[0, :], projection='3d')
    ax0.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.plot_surface(X, T, H_true, cmap=cmap_result, antialiased=True)
    ax0.set_title('Entire Domain', fontdict={'fontsize': 14, 'fontweight': 'medium'})
    ax0.set_xlabel('x', fontdict={'fontsize': 12})
    ax0.set_ylabel('t', fontdict={'fontsize': 12})
    ax0.set_zlabel('|u(x,t)|', fontdict={'fontsize': 12})
    ax0.view_init(30, 45)

    # 2D plots for the slices
    slices = [75, 100, 125]
    colors = ['purple', 'purple', 'purple']  # Added purple for the smallest plot
    for i, slice_idx in enumerate(slices):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x, H_true[slice_idx, :], colors[i], linewidth=2, label='DataSet')
        ax.plot(x, u_pred[slice_idx, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(x,t)$')
        ax.set_title('$t = %.2f$' % (t[slice_idx]), fontsize=10)
        ax.axis('square')
        ax.set_xlim([-5.1, 5.1])
        ax.set_ylim([-0.1, 5.1])
        if i == 1:  # Only add legend to the middle plot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    plt.tight_layout()
    return fig

def plot_results0(u_pred, u_DataSet, x, t, x0, tb, lb, ub, x_f, t_f):
    plt.ioff()

    # Create the grid of the domine
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(X_star, u_DataSet[:, 0], (X, T), method='cubic')

    cmap_result = 'YlGnBu'

    # Data - boundaries
    X0 = tf.stack([x0, 0 * x0], axis=1)  # (x0, 0)
    X_lb = tf.stack([0 * tb + lb[0], tb], axis=1)  # (lb[0], tb)
    X_ub = tf.stack([0 * tb + ub[0], tb], axis=1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb[:, :, 0], X_ub[:, :, 0]])

    # Data - inside
    X_f = tf.stack([x_f, t_f], axis=1)

    ###########  u(x,t)  ##################

    fig1, ax1 = plt.subplots(1, 1)

    # Select first row of the figure
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax1 = plt.subplot(gs0[:, :])

    # Plot data - interior domain
    #        ax1.plot(X_f[:,1], X_f[:,0], 'rx', label = 'Data (initerior) (%d points)' % (X_f.shape[0]), markersize = 1, alpha=.5, clip_on = False)

    h = ax1.imshow(u_pred.T, interpolation='nearest', cmap=cmap_result,
                   extent=[lb[1], ub[1], lb[0], ub[0]],
                   origin='lower', aspect='auto')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(h, cax=cax)

    # Plot data - boundaries
    ax1.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
             clip_on=False)

    # Print lines corresponding to the time slices
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax1.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax1.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax1.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)

    # Title and labels
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax1.set_title('$u(x,t)$', fontsize=10)

    ########   u(x,t) slices ##################

    # Select second row of the figure
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs1[:, :])

    # First slice
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, H_true[75, :], 'b-', linewidth=2, label='DataSet')
    ax.plot(x, u_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    # Second slice
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, H_true[100, :], 'b-', linewidth=2, label='DataSet')
    ax.plot(x, u_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    # Third slice
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, H_true[125, :], 'b-', linewidth=2, label='DataSet')
    ax.plot(x, u_pred[125, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    return fig1

def plot_results1(u_pred, u_DataSet, x, t, x0, tb, lb, ub, x_f, t_f):
    plt.ioff()

    # Create the grid of the domain
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(X_star, u_DataSet[:, 0], (X, T), method='cubic')

    cmap_result = 'viridis'

    # Data - boundaries
    X0 = tf.stack([x0, 0 * x0], axis=1)  # (x0, 0)
    X_lb = tf.stack([0 * tb + lb[0], tb], axis=1)  # (lb[0], tb)
    X_ub = tf.stack([0 * tb + ub[0], tb], axis=1)  # (ub[0], tb)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    # 3D plot for the entire domain
    ax0 = fig.add_subplot(gs[0, :], projection='3d')
    ax0.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax0.plot_surface(X, T, H_true, cmap=cmap_result, antialiased=True)
    ax0.set_title('Entire Domain', fontdict={'fontsize': 14, 'fontweight': 'medium'})
    ax0.set_xlabel('x', fontdict={'fontsize': 12})
    ax0.set_ylabel('t', fontdict={'fontsize': 12})
    ax0.set_zlabel('|u(x,t)|', fontdict={'fontsize': 12})
    ax0.view_init(30, 45)

    # Select second row of the figure
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs1[:, :])

    # First slice
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, H_true[75, :], 'b-', linewidth=2, label='DataSet')
    ax.plot(x, u_pred[75, :], 'r--', linewidth=2, label='Prediction')

    # 2D plot for the first slice
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x, H_true[75, :], 'b-o', linewidth=2, markersize=4, label='DataSet')
    ax1.plot(x, u_pred[75, :], 'r-.', linewidth=2, markersize=4, label='Prediction')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$|u(x,t)|$')
    ax1.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax1.axis('square')
    ax1.set_xlim([-5.1, 5.1])
    ax1.set_ylim([-0.1, 5.1])

    # 2D plot for the second slice
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, H_true[100, :], 'b-^', linewidth=2, markersize=4, label='DataSet')
    ax2.plot(x, u_pred[100, :], 'r:', linewidth=2, markersize=4, label='Prediction')
    ax2.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$|u(x,t)|$')
    ax2.axis('square')
    ax2.set_xlim([-5.1, 5.1])
    ax2.set_ylim([-0.1, 5.1])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    # 2D plot for the third slice
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, H_true[125, :], 'b-*', linewidth=2, markersize=4, label='DataSet')
    ax3.plot(x, u_pred[125, :], 'r--', linewidth=2, markersize=4, label='Prediction')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$|u(x,t)|$')
    ax3.set_xlim([-5.1, 5.1])
    ax3.set_ylim([-0.1, 5.1])
    ax3.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    plt.tight_layout()
    return fig

def plot_error(u_pred, u_DataSet, x, t, x0, tb, lb, ub):
    plt.ioff()

    # Create the grid of the domain
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(X_star, u_DataSet[:, 0], (X, T), method='cubic')
    H_error = np.abs(u_pred - H_true)

    cmap_error = 'inferno'
    cmap_result = 'YlGnBu'

    # Data - boundaries
    X0 = tf.stack([x0, 0 * x0], axis=1)  # (x0, 0)
    X_lb = tf.stack([0 * tb + lb[0], tb], axis=1)  # (lb[0], tb)
    X_ub = tf.stack([0 * tb + ub[0], tb], axis=1)  # (ub[0], tb)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2)

    # 3D plot for the prediction
    ax1 = fig.add_subplot(gs[0], projection='3d', facecolor='lightgray')
    ax1.plot_surface(X, T, u_pred, facecolors=plt.cm.viridis(u_pred), cmap=cmap_result, antialiased=True)
    ax1.set_title('Prediction', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t', fontsize=10)
    ax1.set_ylabel('x', fontsize=10)
    ax1.set_zlabel('|u(x,t)|', fontsize=10)
    ax1.view_init(30, 45)

    # 3D plot for the error
    ax2 = fig.add_subplot(gs[1], projection='3d', facecolor='lightgray')
    ax2.plot_surface(X, T, H_error, facecolors=plt.cm.inferno(H_error), cmap=cmap_error, antialiased=True)
    ax2.set_title('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t', fontsize=10)
    ax2.set_ylabel('x', fontsize=10)
    ax2.set_zlabel('Error', fontsize=10)
    ax2.view_init(30, 45)

    plt.tight_layout()
    return fig

def plot_error0(u_pred, u_DataSet, x, t, x0, tb, lb, ub):
    plt.ioff()

    # Create the grid of the domain
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(X_star, u_DataSet[:, 0], (X, T), method='cubic')
    H_error = np.abs(u_pred - H_true)

    cmap_error = 'inferno'

    cmap_result = 'YlGnBu'

    # Data - boundaries
    X0 = tf.stack([x0, 0 * x0], axis=1)  # (x0, 0)
    X_lb = tf.stack([0 * tb + lb[0], tb], axis=1)  # (lb[0], tb)
    X_ub = tf.stack([0 * tb + ub[0], tb], axis=1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb[:, :, 0], X_ub[:, :, 0]])

    ###########  u(x,t)  ##################

    fig1, ax1 = plt.subplots(1, 1)

    # Select first row of the figure
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.06, left=0.15, right=0.85, wspace=0)
    ax1 = plt.subplot(gs0[:, :])

    h = ax1.imshow(u_pred.T, interpolation='nearest', cmap=cmap_result,
                   extent=[lb[1], ub[1], lb[0], ub[0]],
                   origin='lower', aspect='auto')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(h, cax=cax)

    # Plot data - boundaries
    ax1.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
             clip_on=False)

    ax1.set_ylabel('$x$')
    ax1.set_title('Prediction', fontsize=12)

    # Select first row of the figure
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 2 - 0.04, bottom=0.08, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs1[:, :])

    h1 = ax.imshow(H_error.T, interpolation='nearest', cmap=cmap_error,
                   extent=[lb[1], ub[1], lb[0], ub[0]],
                   origin='lower', aspect='auto')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig1.colorbar(h1, cax=cax)

    # Title and labels
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Absolute error', fontsize=12)

    return fig1

def plot_results_3d(u_pred, u_DataSet, x, t):
    X, T = np.meshgrid(x, t)
    u_pred = griddata(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), u_DataSet[:, 0], (X, T), method='cubic')

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, T, u_pred, cmap='viridis')
    ax1.set_title('Predicted - u(x,t)', fontsize=16)
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('t', fontsize=14)
    ax1.set_zlabel('h', fontsize=14)
    ax1.view_init(30, 45)  # Adjust view angle for better visualization

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, T, H_true, cmap='viridis')
    ax2.set_title('Dataset - u(x,t)', fontsize=16)
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('t', fontsize=14)
    ax2.set_zlabel('h', fontsize=14)
    ax2.view_init(30, 45)  # Adjust view angle for better visualization

    return fig

def plot_error_heatmap(u_pred, u_DataSet, x, t):
    X, T = np.meshgrid(x, t)
    u_pred = griddata(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), u_pred[:, 0], (X, T), method='cubic')
    H_true = griddata(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), u_DataSet[:, 0], (X, T), method='cubic')
    H_error = np.abs(u_pred - H_true)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(H_error, ax=ax, cmap='inferno', cbar_kws={'label': 'Error Magnitude'})
    ax.set_title('Absolute Error Heatmap')
    ax.set_xlabel('x')
    ax.set_ylabel('t')

    return fig

def plot_loss_history(Adam_loss_hist, LBFGS_loss_hist):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    Adam_its = len(Adam_loss_hist)
    Adam_x_axis = range(Adam_its)

    LBFGS_its = len(LBFGS_loss_hist)
    LBFGS_x_axis = range(Adam_its - 1, Adam_its + LBFGS_its - 1, 1)

    plt.plot(Adam_x_axis, Adam_loss_hist, 'r-', linewidth=2, label='Adam optimization')
    plt.plot(LBFGS_x_axis, LBFGS_loss_hist, 'b--', linewidth=2, label='LBFGS optimization')

    plt.legend(fontsize=12)
    plt.title('Loss History Along Iterations', fontsize=16)
    plt.ylabel('Loss Value', fontsize=14)
    plt.xlabel('Total Iteration Number', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    return plt.gcf()
