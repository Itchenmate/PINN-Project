#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""
#%%
import sys
import tensorflow as tf
sys.path.insert(0, 'Utils/')
from Utils.NeurlNet import CustomNeuralNetwork, PINN
#%%

class HeatPINN(PINN):
    def __init__(self, x0, u0, coeffs, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, layers):
        super(HeatPINN, self).__init__()

        # Network architecture
        self.model = CustomNeuralNetwork(ub, lb, layers)

        # Data initialization
        self.x0 = x0
        self.t0 = 0 * x0
        self.u0 = u0
        self.coeffs = coeffs
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star

    def f(self, u, u_t, u_xx,alpha=0.01):
        # f = u_t - alpha * u_xx
        return u_t -alpha*u_xx

    def loss(self):
        x0 = self.x0
        t0 = self.t0
        u0 = self.u0

        # Loss from supervised learning (at t=0)
        coeffs_tiled = tf.reshape(tf.tile(self.coeffs, [1, tf.shape(x0)[0]]), [-1, 6])
        X0 = tf.concat([x0, t0, coeffs_tiled], axis=1)
        u_pred = self.model(X0)
        loss_0 = tf.reduce_mean(tf.square(u0 - u_pred))

        # Loss from heat equation constraint (at the anchor pts)
        f_val = self.f_nn()
        loss_f = tf.reduce_mean(tf.square(f_val))

        # Loss from boundary conditions
        u_lb_pred, u_x_lb_pred = self.u_grad(self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.u_grad(self.x_ub, self.t_ub)

        loss_b = tf.reduce_mean(tf.square(u_lb_pred))+ tf.reduce_mean(tf.square(u_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred))+ tf.reduce_mean(tf.square(u_x_ub_pred))

        return loss_0 + loss_f + loss_b

    def u_grad(self, x, t):
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(t)
            coeffs_tiled = tf.tile(self.coeffs, [1, tf.shape(x)[0]])
            coeffs_tiled = tf.transpose(coeffs_tiled)
            X = tf.concat([x, t, coeffs_tiled], axis=1)
            u = self.model(X)

        u_x = tape.gradient(u, x)

        return u, u_x

    def f_nn(self):
        x_f = self.x_f
        t_f = self.t_f

        with tf.GradientTape(persistent=True) as tape:
            x_f = tf.convert_to_tensor(self.x_f)  # Convert to TensorFlow tensor
            t_f = tf.convert_to_tensor(self.t_f)  # Convert to TensorFlow tensor
            tape.watch(x_f)
            tape.watch(t_f)
            coeffs_tiled = tf.reshape(tf.tile(self.coeffs, [1, tf.shape(x_f)[0]]), [-1, 6])
            X_f = tf.concat([x_f, t_f, coeffs_tiled], axis=1)
            u = self.model(X_f)
            u_t = tape.gradient(u, t_f)
            u_x = tape.gradient(u, x_f)

        u_xx = tape.gradient(u_x, x_f)
        del tape

        f_val = self.f(u, u_t, u_xx)
        return f_val

    def predict(self, x, t):
        x = tf.reshape(x, [-1, 1])
        t = tf.reshape(t, [-1, 1])
        coeffs_tiled = tf.tile(self.coeffs, [1, tf.shape(x)[0]])
        coeffs_tiled = tf.transpose(coeffs_tiled)
        X = tf.concat([x, t, coeffs_tiled], axis=1)
        u_pred = self.model(X)
        return u_pred
