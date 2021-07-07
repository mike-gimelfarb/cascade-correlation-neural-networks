import warnings
import numpy as np
import tensorflow.compat.v1 as tf

from cascor.units.unit import CCNNUnit
from cascor._settings import dtype


class BayesianLinear(CCNNUnit):
    
    def __init__(self, alpha=0.01, beta=100.0):
        self.alpha = alpha
        self.beta = beta
    
    def build(self, num_inputs, num_outputs, num_targets):
        if num_outputs != num_targets:
            raise Exception('num. outputs != num. targets: cannot use bayesian linear regression as candidate!')
            
        # initialize Gaussian prior on the weights
        self.mean, self.cov, self.cov_inv = [], [], []
        for each_num_input in num_inputs:
            self.mean.append(np.zeros((each_num_input + 1, num_outputs)))
            self.cov.append(np.eye(each_num_input + 1) / self.alpha)
            self.cov_inv.append(np.linalg.inv(self.cov[-1])) 
 
    def evaluate_losses(self, X, y):
        losses = []
        for mean, x in zip(self.mean, X):
            Phi = np.column_stack([x, np.ones((x.shape[0], 1))])
            loss = np.mean((Phi @ mean - y) ** 2)
            losses.append(loss)
        return np.array(losses)

    def train(self, X, y):
         
        # do posterior update
        for i, x in enumerate(X):
            Phi = np.column_stack([x, np.ones((x.shape[0], 1))])
            new_cov_inv = self.cov_inv[i] + self.beta * Phi.T @ Phi
            self.cov[i] = np.linalg.inv(new_cov_inv)
            self.mean[i] = self.cov[i] @ (self.cov_inv[i] @ self.mean[i] + self.beta * Phi.T @ y)
            self.cov_inv[i] = new_cov_inv            
        return self.evaluate_losses(X, y)
    
    def finalize(self, sess, inputs, head=-1):
        
        # compute predictive distribution
        with sess.graph.as_default():
            Phi = tf.pad(inputs, [[0, 0], [0, 1]], constant_values=1.0)
            y_pred = Phi @ tf.constant(self.mean[head], dtype=dtype)
            y_pred_cov = 1.0 / self.beta + Phi @ tf.constant(self.cov[head], dtype=dtype) @ tf.transpose(Phi)
        return y_pred, y_pred_cov
    
    def clear(self):
        pass
