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

class neural_net_2out(CustomNeuralNetwork):
    def __init__(self, ub, lb, layers):
        super(neural_net_2out, self).__init__(ub, lb, layers)

    def call(self, inputs, training=False):
        output = super(neural_net_2out, self).call(inputs)
        return output[:,0], output[:,1]


class S_PINN(PINN):

    def __init__(self, x0, u0, v0, x_ub, x_lb, t_b, x_f, t_f, X_star, ub, lb, layers):
        super(S_PINN, self).__init__()

        # Network architecture
        self.model = neural_net_2out(ub, lb, layers)
        
        # Data initialization
        self.x0 = x0
        self.t0 = 0*x0
        self.u0 = u0
        self.v0 = v0
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.t_ub = t_b
        self.t_lb = t_b
        self.x_f = x_f
        self.t_f = t_f
        self.X_star = X_star

    def f(self,u,v,u_t,v_t,u_xx,v_xx):
        #f= i*h_t +0.5*h_xx + |h|^2*h , h(x,t) = u(x,t) + i*v(x,t) ==0 ,Lets mutiply by -i and get:
        #f=h_t -0.5*i(h_xx) -i*|h|^2 * h ---> h= u+iv Thus:
        #f = (u_t +0.5*v_xx +(u^2 +v^2) * v ) + i* (v_t -0.5*u_xx -(u^2+v^2) * u)
        f_real = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_imag = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u
        return f_real , f_imag
  
    # Loss definition
    def loss(self):
        x0 = self.x0
        t0 = self.t0
        u0 = self.u0
        v0 = self.v0

        # Loss from PDE condtion (implicit functions)
        X0 = tf.stack([x0, t0], axis=1)
        u_pred, v_pred = self.model(X0)
        loss_0 = tf.reduce_mean(tf.square(u0 - u_pred)) + tf.reduce_mean(tf.square(v0 - v_pred))

        # Loss from initial condtions
        f_real, f_image = self.net_f_uv()
        loss_f = tf.reduce_mean(tf.square(f_real)) + tf.reduce_mean(tf.square(f_image))
          
        # Loss from boundary conditions
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.grad_uv(self.x_lb, self.t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.grad_uv(self.x_ub, self.t_ub)
    
        loss_b = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
             tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))
    
        return loss_0+ loss_b + loss_f

    def grad_uv(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1)
            u, v = self.model(X)
        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)

        return u, v, u_x, v_x

    def net_f_uv(self):
        x_f = self.x_f
        t_f = self.t_f
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)
            X_f = tf.stack([x_f, t_f], axis=1)
            u, v = self.model(X_f)
            u_x = tape.gradient(u, x_f)
            v_x = tape.gradient(v, x_f)

        u_t = tape.gradient(u, t_f)
        v_t = tape.gradient(v, t_f)
        u_xx = tape.gradient(u_x, x_f)
        v_xx = tape.gradient(v_x, x_f)
        del tape
        f_real, f_imag = self.f(u, v, u_t, v_t, u_xx, v_xx)

        return f_real, f_imag

    def predict(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.stack([x, t], axis=1)
            u, v = self.model(X)
            u_x = tape.gradient(u, x)
            v_x = tape.gradient(v, x)
        u_t = tape.gradient(u, t)
        v_t = tape.gradient(v, t)
        u_xx = tape.gradient(u_x, x)
        v_xx = tape.gradient(v_x, x)
        del tape
        f_real , f_imag = self.f(u,v,u_t,v_t,u_xx,v_xx)

        return u, v, f_real, f_imag



