#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Gadi & Aviv Burshtein
"""

import tensorflow as tf
import scipy.optimize
import time
import numpy as np

#%%

class CustomNeuralNetwork(tf.keras.Sequential):
    """
    A custom neural network class that inherits from tf.keras.Sequential.
    This class is designed to represent a neural network with specific architecture and methods.
    """

    def __init__(self, upper_bound, lower_bound, layers):
        super(CustomNeuralNetwork, self).__init__()

        # Ensure float64 for Keras backend
        tf.keras.backend.set_floatx('float64')

        self.t_last_callback = 0

        ##PDE_HEAT:
        self.lower_bound = tf.convert_to_tensor([lower_bound, lower_bound] + [0.0] * 6, dtype=tf.float32)
        self.upper_bound = tf.convert_to_tensor([upper_bound, upper_bound] + [1.0] * 6, dtype=tf.float32)

        ## PDE_9 AND PDE_AC:
        """# Store normalization bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound"""

        # Define the network architecture
        self._build_network_architecture(layers)

        # Compute sizes for weights and biases for future use
        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))
#PDE_HEAT:
    def _build_network_architecture(self, layers):
        # Define the architecture of the neural network
        # input layers:
        self.add(tf.keras.layers.InputLayer(input_shape=(8,)))

        # Lambda layer for normalization
        self.add(tf.keras.layers.Lambda(
            lambda X: 2.0 * (tf.cast(X, tf.float32) - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0))

        # Store the input shapes for each layer
        self.input_shapes = [layers[0]]

        # hidden layers
        for width in layers[1:-1]:
            self.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="orthogonal"))
            self.input_shapes.append(width)

        # output layers
        self.add(tf.keras.layers.Dense(layers[-1], activation=None, kernel_initializer="orthogonal"))

#PDE_9:
    """def _build_network_architecture(self, layers):
        # Define the architecture of the neural network
        # input layers:
        self.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # lambda layer for normalization
        self.add(tf.keras.layers.Lambda(
            lambda X: 2.0 * (X - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0))
        # hidden layers
        for width in layers[1:-1]:
            self.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="orthogonal"))
        # output layers
        self.add(tf.keras.layers.Dense(layers[-1], activation=None, kernel_initializer="orthogonal"))"""
#PDE_AC :
    """def _build_network_architecture(self, layers):
        # Define the architecture of the neural network
        # input layers:
        self.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))

        # lambda layer for normalization
        self.add(tf.keras.layers.Lambda(
            lambda X: 2.0 * (X - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0))

        # Store the input shapes for each layer
        self.input_shapes = [layers[0]]

        # hidden layers
        for width in layers[1:-1]:
            self.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="orthogonal"))
            self.input_shapes.append(width)

        # output layers
        self.add(tf.keras.layers.Dense(layers[-1], activation=None, kernel_initializer="orthogonal"))"""

    def get_weights(self, convert_to_tensor=True):
        w = []
        for layer in self.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = tf.convert_to_tensor(w)

        # Make the output Fortran contiguous
        w = np.copy(w, order='F')

        return w

    def set_weights(self, w):
        for i, layer in enumerate(self.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]

            # Use the input_shapes list to determine the shape of the weights
            w_div = self.input_shapes[i]

            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    #PDE_9:
    """def set_weights(self, w):
        for i, layer in enumerate(self.layers[1:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)"""
#%%
class PINN():
    """
    Physics-Informed Neural Network (PINN) class.
    Represents a neural network that incorporates physical principles.
    """

    def __init__(self):
        self.iteration_count = 0
        self.LBFGS_max_iterations = 0
        self.LBFGS_hist = []
        self.elapsed = 0
        self.elapsed_l = 0
        self.elapsed_t = 0

    def loss_and_flat_grad(self, w):

        with tf.GradientTape() as tape:
            self.model.set_weights(w)
            loss_value = self.loss()
        grad = tape.gradient(loss_value, self.model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)

        # Make the output Fortran contiguous
        loss_value = np.copy(loss_value, order='F')
        grad_flat = np.copy(grad_flat, order='F')

        return loss_value, grad_flat

    def loss(self):
        raise NotImplementedError("Loss function must be implemented in a subclass or instance.")

    def train(self, Adam_iterations, LBFGS_max_iterations):
        Adam_hist = []
        if Adam_iterations:
            print('♪♪♪ Adam optimization ♪♪♪')
            optimizer = tf.keras.optimizers.Adam()
            start_time = time.time()
            iteration_start_time = start_time
            for i in range(Adam_iterations):
                current_loss = self.Adam_train_step(optimizer)
                Adam_hist.append(current_loss)
                iteration_time = str(time.time() - iteration_start_time)[:5]
                print(f'Loss: {current_loss.numpy()}, time: {iteration_time}, iter: {i+1}/{Adam_iterations}')
                iteration_start_time = time.time()
            self.elapsed = time.time() - start_time
            Adam_hist.append(self.loss())
            print(f'Adam optimization training time: {self.elapsed:.4f}')
            start_time_l = time.time()
            self.start_time_l_iter=start_time_l
            self.LBFGS_max_iterations=LBFGS_max_iterations
            self.LBFGS_hist = [self.loss()]
            if LBFGS_max_iterations:
                print('♫♫♫ L-BFGS optimization ♫♫♫')
                maxiter = LBFGS_max_iterations
                self.t_last_callback = time.time()
                results = scipy.optimize.minimize(self.loss_and_flat_grad,
                                                  self.model.get_weights(),
                                                  method='L-BFGS-B',
                                                  jac=True,
                                                  callback=self.callback,
                                                  options={'maxiter': maxiter,
                                                           'maxfun': 50000,
                                                           'maxcor': 50,
                                                           'maxls': 50,
                                                           'ftol': 1.0 * np.finfo(float).eps})
                optimal_w = results.x
                self.model.set_weights(optimal_w)
                self.elapsed_l = time.time() - start_time_l
                self.elapsed_t  = time.time() - start_time
                print(f'L-BFGS optimization training time: {self.elapsed_l:.4f}')
                print(f'Total training time: {self.elapsed_t:.4f}')
                print(f'☺☺☺ The Model Has Trained ☺☺☺\nFinal loss: {self.loss().numpy()}')

            optimal_w = results.x
            self.model.set_weights(optimal_w)

        return Adam_hist, self.LBFGS_hist, [self.elapsed, self.elapsed_l,self.elapsed_t,self.loss().numpy()]

    @tf.function
    def Adam_train_step(self, optimizer):

        with tf.GradientTape() as tape:
            loss_value = self.loss()

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value

    def callback(self,*args):
        self.iteration_count += 1
        iteration_time = str(time.time() - self.start_time_l_iter)[:5]
        self.start_time_l_iter  = time.time()
        loss_value = self.loss().numpy()
        self.LBFGS_hist.append(loss_value)
        print(f'Loss: {loss_value}, time: {iteration_time}, iter: {self.iteration_count}/{self.LBFGS_max_iterations}')
