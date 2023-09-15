#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""

import sys
sys.path.insert(0, 'Utils/')
from Utils.NeurlNet import CustomNeuralNetwork, PINN
import tensorflow as tf


#%%

class AC_PINN(PINN):
    
    def __init__(self, x0, u0, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, layers):
        self.iteration_count = 0
        super(PINN, self).__init__()
        
        # Network architecture
        self.model = CustomNeuralNetwork(ub, lb, layers)
        
        # Data initialization
        self.x0 = x0
        self.t0 = 0*x0
        self.u0 = u0
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star
        
    # Loss definition
    def loss(self):
        
        x0 = self.x0
        t0 = self.t0
        u0 = self.u0
    
        # Loss from supervised learning (at t=0)
        X0 = tf.stack([x0, t0], axis=1)
        u_pred = self.model(X0)
        loss_0 = tf.reduce_mean(tf.square(u0 - u_pred))
    
        # Loss from PDE at the collocation pts
        f_u = self.u_grad()
        loss_f = tf.reduce_mean(tf.square(f_u))
          
        # Loss from boundary conditions
        u_lb_pred, u_x_lb_pred = self.grad_only_u(self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.grad_only_u(self.x_ub, self.t_ub)
    
        loss_b = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred))
    
        return loss_b+loss_f+loss_b

    def grad_only_u(self, x, t):
    
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1) # shape = (N_f,2)
        
            u = self.model(X)
         
        u_x = tape.gradient(u, x)

        return u, u_x

    def u_grad(self):
        x_f = self.x_f
        t_f = self.t_f
        with tf.GradientTape(persistent=True) as tape:    
            tape.watch(x_f)
            tape.watch(t_f)
            X_f = tf.stack([x_f, t_f], axis=1) # shape = (N_f,2)
            u = self.model(X_f)
            u_x = tape.gradient(u, x_f)
        u_t = tape.gradient(u, t_f)
        u_xx = tape.gradient(u_x, x_f)
        del tape
        f_u = u_t - 5.0*u + 5.0*u**3 - 0.0001*u_xx
        return f_u

    def predict(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1)  # shape = (N_f,2)
            u = self.model(X)
            u_x = tape.gradient(u, x)

        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
        del tape
        f_u = u_t - 5.0*u + 5.0*u**3 - 0.0001*u_xx

        return u, f_u

