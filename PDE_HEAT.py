#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""

import sys
import tensorflow as tf
sys.path.insert(0, 'Utils/')
from Utils.NeurlNet import CustomNeuralNetwork, PINN

class HeatPINN(PINN):
    def __init__(self, x0, u0, coeffs, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, layers):
        super(HeatPINN, self).__init__()

        # Network architecture
        self.model = CustomNeuralNetwork(ub, lb, layers)

        # Data initialization
        self.x0 = x0
        self.t0 = 0 * x0
        self.u0 = u0
        self.coeffs = tf.convert_to_tensor(coeffs)
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star
        self.coeffs_tiled = tf.reshape(tf.tile(self.coeffs, [1, tf.shape(self.x_f)[0]]), [-1, 6])

    def loss(self):
        x0 = self.x0
        t0 = self.t0
        u0 = self.u0
        # Loss from supervised learning (at t=0)
        coeffs_tiled = tf.reshape(tf.tile(self.coeffs, [1, tf.shape(x0)[0]]), [-1, 6])
        X0 = tf.concat([x0,t0, coeffs_tiled], axis=1)  # Include coefficients in input
        #X0 = tf.stack([x0, t0], axis=1)
        u_pred = self.model(X0)
        loss_0 = tf.reduce_mean(tf.square(u0 - u_pred))

        # Loss from PDE at the collocation pts
        f_u = self.net_f_u()
        loss_f = tf.reduce_mean(tf.square(f_u))

        # Loss from boundary conditions
        u_lb_pred, u_x_lb_pred = self.net_u(self.x_lb, self.t_lb)
        u_ub_pred, u_x_ub_pred = self.net_u(self.x_ub, self.t_ub)

        loss_b = tf.reduce_mean(tf.square(u_lb_pred))+ tf.reduce_mean(tf.square(u_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred))+ tf.reduce_mean(tf.square(u_x_ub_pred))


        return loss_0 + loss_f + loss_b

    def net_f_u(self):
        h = 1e-4
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            self.coeffs_tiled = tf.reshape(tf.tile(self.coeffs, [1, tf.shape(self.x_f)[0]]), [-1, 6])
            X_f = tf.concat([self.x_f, self.t_f, self.coeffs_tiled], axis=1)
            u = self.model(X_f)
            u_x = tape.gradient(u, self.x_f)

            u_x_plus_h = tape.gradient(self.model(tf.concat([self.x_f + h, self.t_f, self.coeffs_tiled], axis=1)),
                                       self.x_f)
            u_x_minus_h = tape.gradient(self.model(tf.concat([self.x_f - h, self.t_f, self.coeffs_tiled], axis=1)),
                                        self.x_f)
            u_xx = (u_x_plus_h - u_x_minus_h) / (2 * h)

        u_t = tape.gradient(u, self.t_f)
        del tape
        f_val = u_t - u_xx
        return f_val

    def predict(self, x, t):
        x = tf.reshape(x, [-1, 1])
        t = tf.reshape(t, [-1, 1])
        coeffs_tiled = tf.tile(self.coeffs, [1, tf.shape(x)[0]])
        coeffs_tiled = tf.transpose(coeffs_tiled)
        X = tf.concat([x, t, coeffs_tiled], axis=1)
        u_pred = self.model(X)
        return u_pred

    def net_u(self, x, t):
        coeffs = self.coeffs  # Input coefficients

        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(t)
            tape.watch(coeffs)  # Watch coefficients
            print("Shape of x:", tf.shape(x))
            print("Shape of coeffs:", tf.shape(coeffs))
            multiplier = tf.shape(x)[0] // tf.shape(coeffs)[0]
            print("Tiling multiplier:", multiplier)
            coeffs_tiled = tf.tile(coeffs, [multiplier, 1])
            X = tf.concat([x, t, coeffs_tiled], axis=1)
            u = self.model(X)

        u_x = tape.gradient(u, x)

        return u, u_x

