#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""
#%%
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import scipy.io
from pyDOE import lhs 
import sys
import matplotlib.pyplot as plt
#settings of the directory
sys.path.insert(0, 'Utils/')

#import from our classes:
from PDE_AC import AC_PINN
from Utils.plotting_AC import plot_results, plot_error, plot_loss_history, plot_error_heatmap, plot_results_3d



#%%

if __name__ == "__main__":

    # Some usueful set-up in order  to ensure that the results of your code are reproducible:
    np.random.seed(5890)
    tf.random.set_seed(5890)
    
    # Domain bounds
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])

    # Cons. number of pts of the training
    N0 = 50  # Number of training pts from x=0
    N_b = 50  # Number of training pts from the boundaries
    N_f = 20000  # Number of training pts from the PDE(Implicit function)

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
    # %%
    # Import data
    data = scipy.io.loadmat('Data/AC.mat')

    t = data['tt'].flatten()[:,None] # T x 1
    x = data['x'].flatten()[:,None] # N x 1
    Exact = np.real(data['uu']).T # T x N
    
    # x is too big to be handled (512) so it is reduced by half (to 256)
    # Only even entries are taken
    x = x[ np.arange(0,x.shape[0],2) ,:]
    Exact = Exact[:, np.arange(0,Exact.shape[1],2)]
    
    # Creation of the 2D domain
    X, T = np.meshgrid(x,t)
    
    # The whole domain flattened, on which the final prediction will be made
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    
    # Choose N0 training points from x and the corresponding u, v at t=0
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact.T[idx_x,0:1]
    
    # Choose N_b training points from the time
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    # Latin Hypercube Sampling of N_f points from the interior domain
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # 2D locations on the domain of the boundary training points
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

    # Initial condition pts
    x0 = X0[:,0:1]
    t0 = X0[:,1:2]

    # Boundary pts used for constraint
    x_lb = X_lb[:,0:1]
    t_lb = X_lb[:,1:2]

    x_ub = X_ub[:,0:1]
    t_ub = X_ub[:,1:2]
        
    # Anchor pts for supervised learning
    x_f = X_f[:,0:1]
    t_f = X_f[:,1:2]    
    
    # All these are numpy.ndarray with dtype float64
    
    # Conversion to tensors. Recall to WATCH inside a tape
    x0 = tf.convert_to_tensor(x0[:,0])
    t0 = tf.convert_to_tensor(t0[:,0])
    u0 = tf.convert_to_tensor(u0[:,0])
    x_lb = tf.convert_to_tensor(x_lb[:,0])
    t_lb = tf.convert_to_tensor(t_lb[:,0])
    x_ub = tf.convert_to_tensor(x_ub[:,0])
    t_ub = tf.convert_to_tensor(t_ub[:,0])
    x_f = tf.convert_to_tensor(x_f[:,0])
    t_f = tf.convert_to_tensor(t_f[:,0])
    X_star = tf.convert_to_tensor(X_star)

    # %%
    layers = [2,50,50,50,1]
    model = AC_PINN(x0, u0, x_ub, x_lb, t_ub, x_f, t_f, X_star, ub, lb, layers)

#%%

    # %%
    #####################################################
    #                                                   #
    #          MODEL TRAINING AND PREDICTION            #
    #                                                   #
    #####################################################
    """
    TTTT RRRR    A   IIIII N   N IIIII N   N               A   N   N DDD         PPPP  RRRR  EEEEE DDD   IIIII       TTTTT IIIII  OOO  N   N 
      T   R   R  A A    I   NN  N   I   NN  N              A A  NN  N D  D        P   P R   R E     D  D    I           T     I   O   O NN  N 
      T   RRRR  AAAAA   I   N N N   I   N N N             AAAAA N N N D   D       PPPP  RRRR  EEEEE D   D   I           T     I   O   O N N N 
      T   R R   A   A   I   N  NN   I   N  NN             A   A N  NN D  D        P     R R   E     D  D    I           T     I   O   O N  NN 
      T   R  RR A   A IIIII N   N IIIII N   N             A   A N   N DDD         P     R  RR EEEEE DDD   IIIII         T   IIIII  OOO  N   N 
    """

    adam_iterations = 1000  # Number of training steps
    lbfgs_max_iterations = 2000 # Max iterations for lbfgs
    
    Adam_hist, LBFGS_hist, elpases = model.train(adam_iterations, lbfgs_max_iterations)
        
    
#%%    
        
    # final prediction
    u_pred = model.model(X_star)
                
    # final error
    u_pred = tf.reshape(u_pred, shape=(51456,1))

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))


#%%

    # Plotting
    # Ensure the directory 'figures' exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Plotting (res ,pred and loss )
    fig_res = plot_results_3d(u_pred, u_star, x, t)
    fig_err = plot_error_heatmap(u_pred, u_star, x,t)
    fig_los_his = plot_loss_history(Adam_hist, LBFGS_hist)

    # Save the figures
    fig_res.savefig('figures/fig_res_3d.png')
    fig_err.savefig('figures/fig_err_3d.png')
    fig_los_his.savefig('figures/fig_loss_history.png')

    # Plotting (res and pred) OLD
    fig_res = plot_results(u_pred, u_star, x,t,x0,tb,lb,ub, x_f,t_f)
    fig_err = plot_error(u_pred, u_star, x,t,x0,tb,lb,ub)

    # Save the figures
    fig_res.savefig('figures/fig_res.png')
    fig_err.savefig('figures/fig_err.png')
#%%
    #Arrange in txt and pdf:
    #style:
    color_palette = {
        'title': '#2C3E50',
        'text': '#34495E',
        'highlight': '#E74C3C',
        'secondary': '#7F8C8D',
        'background': '#ECF0F1'
    }
    # Set global font settings for a professional look
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.titleweight'] = 'bold'

    def save_training_summary(ax, elpases, lbfgs_max_iterations):
        ax.text(0.1, 0.9, f'Adam Training Time: {elpases[0]}', transform=ax.transAxes, color=color_palette['text'])
        ax.text(0.1, 0.8, f'LBFGS Training Time: {elpases[1]}', transform=ax.transAxes, color=color_palette['text'])
        ax.text(0.1, 0.7, f'Total Training Time: {elpases[2]}', transform=ax.transAxes, color=color_palette['text'])
        ax.text(0.1, 0.5, f'Adam Iterations: {adam_iterations}', transform=ax.transAxes, color=color_palette['secondary'])
        ax.text(0.1, 0.4, f'L-BFGS Iterations: {lbfgs_max_iterations}', transform=ax.transAxes, color=color_palette['secondary'])
        ax.set_title('Training Summary', fontsize=18, color=color_palette['title'])
        ax.axis('off')


    def save_model_summary(ax, error_u):
        ax.text(0.1, 0.8, f'Error u: {error_u}', transform=ax.transAxes, color=color_palette['text'])
        ax.text(0.1, 0.2, f'Final Loss: {elpases[3]}', transform=ax.transAxes, color=color_palette['highlight'])
        ax.set_title('Model Error Summary', fontsize=16, color=color_palette['title'])
        ax.axis('off')


    with PdfPages('figures/summary.pdf') as pdf:
        # Training Summary
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=color_palette['background'])
        save_training_summary(ax, elpases, lbfgs_max_iterations)
        pdf.savefig(facecolor=color_palette['background'])
        plt.close()

        # Model Summary
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=color_palette['background'])
        save_model_summary(ax, error_u)
        pdf.savefig(facecolor=color_palette['background'])
        plt.close()

        # Other plots
        fig_res = plot_results_3d(u_pred, u_star, x, t)
        pdf.savefig(fig_res)
        plt.close()

        fig_err = plot_error_heatmap(u_pred, u_star, x, t)
        pdf.savefig(fig_err)
        plt.close()

        fig_los_his2 = plot_loss_history(Adam_hist, LBFGS_hist)
        pdf.savefig(fig_los_his)
        plt.close()


#%%
#make a csv file for future analysis:
h_pred = np.array(u_pred).flatten()
h_star = np.array(u_star).flatten()
x = np.array(x).flatten()
t = np.array(t).flatten()
x0 = np.array(x0).flatten()
tb = np.array(tb).flatten()
lb = np.array(lb).flatten()  # Assuming lb is a list/array, not a single value
ub = np.array(ub).flatten()  # Assuming ub is a list/array, not a single value
x_f = np.array(x_f).flatten()
t_f = np.array(t_f).flatten()
Adam_hist = np.array(Adam_hist).flatten()
LBFGS_hist = np.array(LBFGS_hist).flatten()

# Find the maximum length among all arrays
max_len = max(len(h_pred), len(h_star), len(x), len(t), len(x0), len(tb), len(lb), len(ub), len(x_f), len(t_f), len(Adam_hist), len(LBFGS_hist))

# Create a dataframe
df = pd.DataFrame({
    'h_pred': pd.Series(h_pred),
    'h_star': pd.Series(h_star),
    'x': pd.Series(x),
    't': pd.Series(t),
    'x0': pd.Series(x0),
    'tb': pd.Series(tb),
    'lb': pd.Series(lb),
    'ub': pd.Series(ub),
    'x_f': pd.Series(x_f),
    't_f': pd.Series(t_f),
    'Adam_hist': pd.Series(Adam_hist),
    'LBFGS_hist': pd.Series(LBFGS_hist),
    'Final_Loss': pd.Series([elpases[3]]*max_len)
})

# Save the dataframe to a CSV file
df.to_csv('figures/data.csv', index=False)

