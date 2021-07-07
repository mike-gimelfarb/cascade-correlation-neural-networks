import time
import itertools
import numpy as np
import scipy.optimize as opt
import tensorflow.compat.v1 as tf  # if using tensorflow version >= 2

from cascor.units.unit import CCNNUnit
from cascor._settings import dtype


class TensorflowPerceptron(CCNNUnit):
    
    def __init__(self, activations, loss_function, stopping_rule, optimizer=tf.train.AdamOptimizer,
                 optimizer_args={}, batch_size=32, regularizer=None, reg_penalty=0.0,
                 weight_init=tf.glorot_uniform_initializer(), print_freq=100):
        self.activations = activations
        self.loss_function = loss_function
        self.stopping_rule = stopping_rule
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.reg_penalty = reg_penalty
        self.weight_init = weight_init
        self.print_freq = print_freq
        
    @staticmethod
    def _build_perceptron(inputs, num_outputs, activation, weight_init):
        
        # initialize weights and biases
        num_inputs = inputs.get_shape()[-1]
        w = tf.Variable(weight_init(shape=[num_inputs, num_outputs], dtype=dtype), name='w')
        b = tf.Variable(np.zeros((num_outputs,)), dtype=dtype, name='b')
        weights = [w, b]
        
        # initialize prediction
        y_pred = activation(inputs @ w + b)
        if isinstance(y_pred, tuple):
            y_pred, extra_vars = y_pred
            weights.append(extra_vars)
        return y_pred, weights
        
    def build(self, num_inputs, num_outputs, num_targets):
        
        # build a new graph and session in which training will take place
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default(): 
            
            # variables common to all neurons    
            self.targets = tf.placeholder(dtype=dtype, shape=[None, num_targets], name='targets')
            self.inputs, self.losses, self.neurons, self.train_ops, variables = [], [], [], [], []
            
            # group neurons based on their inputs
            for each_num_inputs in num_inputs:
                inputs = tf.placeholder(dtype=dtype, shape=[None, each_num_inputs], name='inputs')
                self.inputs.append(inputs)
                
                # group neurons based on their activations
                for activation in self.activations:
                    
                    # construct prediction of neuron
                    y_pred, weights = self._build_perceptron(inputs, num_outputs, activation, self.weight_init)
                    self.neurons.append((weights, activation))
                    
                    # construct loss function
                    loss = reg_loss = self.loss_function(self.targets, y_pred)
                    self.losses.append(loss)
                    if self.regularizer is not None:
                        reg_loss = reg_loss + self.reg_penalty * self.regularizer(weights[0])
                    
                    # construct optimizer
                    sgd = self.optimizer(**self.optimizer_args)
                    self.train_ops.append(sgd.minimize(reg_loss, var_list=weights))
                    variables.extend(weights + sgd.variables()) 
                    
            # group training ops and initialize variables  
            self.train_ops = tf.group(self.train_ops)
            self.sess.run(tf.variables_initializer(variables))
    
    def _run(self, ops, X, y):
        return self.sess.run(ops, feed_dict={**dict(zip(self.inputs, X)), self.targets: y})
        
    def evaluate_losses(self, X, y):
        return np.array(self._run(self.losses, X, y))

    @staticmethod
    def _batch_iterator(num_patterns, batch_size):
        batch_size = min(batch_size, num_patterns)
        indices = np.arange(num_patterns)
        np.random.shuffle(indices)
        i = 0
        while i + batch_size <= num_patterns:
            ip = i
            i += batch_size
            yield indices[ip:i]

    def train(self, X, y):
        self.stopping_rule.initialize()
        start_time = time.time()
        for t in itertools.count():
            
            # use gradient descent
            for batch in self._batch_iterator(y.shape[0], self.batch_size):
                self._run(self.train_ops, [x[batch] for x in X], y[batch])
        
            # evaluate loss
            losses = self.evaluate_losses(X, y)
            best_loss = np.min(losses)
            best_index = np.argmin(losses)
            if t % self.print_freq == 0:
                print('epoch {0} \t best loss {1:.8f} \t best neuron {2}'.format(t, best_loss, best_index))
            
            # check if output weights have converged
            if self.stopping_rule.update(best_loss): 
                break
        
        # print the final performance after training all output neurons  
        end_time = 1000. * (time.time() - start_time) / float(t)    
        print('epochs {}, average time/epoch {:.6f} ms, losses {}'.format(t, end_time, losses))
        return losses
    
    def finalize(self, sess, inputs, head=-1):
        weights, activation = self.neurons[head]
        weights = self.sess.run(weights)
        with sess.graph.as_default():
            w, b = tf.constant(weights[0]), tf.constant(weights[1])
            y_pred = activation(inputs @ w + b)
            if isinstance(y_pred, tuple):
                y_pred, extra_vars = y_pred
                sess.run(extra_vars.assign(weights[2]))
        return y_pred, w, b

    def clear(self):
        self.sess.close()
        del self.targets, self.inputs, self.losses, self.neurons, self.train_ops
        del self.graph


class ScipyPerceptron(TensorflowPerceptron):
    
    def __init__(self, activations, loss_function,
                 optimizer_args={'method': 'bfgs', 'options': {'disp': True}}, regularizer=None,
                 reg_penalty=0.0, weight_init=tf.glorot_uniform_initializer()):
        self.activations = activations
        self.loss_function = loss_function
        self.optimizer_args = optimizer_args
        self.regularizer = regularizer
        self.reg_penalty = reg_penalty
        self.weight_init = weight_init
        
    @staticmethod
    def _build_assignable_perceptron(inputs, num_outputs, activation, weight_init):
        y_pred, weights = TensorflowPerceptron._build_perceptron(inputs, num_outputs, activation, weight_init)
        
        # allow for assignment of variables
        placeholders, assign_ops = [], []
        for variable in weights:
            placeholder = tf.placeholder(dtype=dtype, shape=variable.get_shape())
            assign_op = tf.assign(variable, placeholder)
            placeholders.append(placeholder)
            assign_ops.append(assign_op)
        return y_pred, weights, placeholders, assign_ops
        
    def build(self, num_inputs, num_outputs, num_targets):
        
        # build a new graph and session in which output neuron training will take place
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default(): 
    
            # variables common to all neurons   
            self.targets = tf.placeholder(dtype=dtype, shape=[None, num_targets], name='targets')
            self.inputs, self.neurons, self.losses, self.train_ops, self.place_and_assign, variables = \
                [], [], [], [], [], []
                    
            # group neurons based on their inputs
            for each_num_inputs in num_inputs:
                inputs = tf.placeholder(dtype=dtype, shape=[None, each_num_inputs], name='inputs')
                self.inputs.append(inputs)
                
                # group neurons based on their activations
                for activation in self.activations:
                    
                    # construct prediction of output neuron
                    y_pred, weights, placeholders, assign_ops = self._build_assignable_perceptron(
                        inputs, num_outputs, activation, self.weight_init)
                    self.neurons.append((weights, activation))
                    self.place_and_assign.append((placeholders, assign_ops))
            
                    # construct loss function
                    loss = reg_loss = self.loss_function(self.targets, y_pred)
                    self.losses.append(loss)
                    if self.regularizer is not None:
                        reg_loss = reg_loss + self.reg_penalty * self.regularizer(weights[0])
                    
                    # construct optimizer
                    self.train_ops.append((reg_loss, tf.gradients(reg_loss, weights)))
                    variables.extend(weights)
            
            # initialize variables
            self.sess.run(tf.variables_initializer(variables))

    def train(self, X, y):
        
        # train each neuron sequentially
        for x, train_op, (placeholders, assign_ops), (weights, _) in zip(
            X, self.train_ops, self.place_and_assign, self.neurons):
            
            # objective function for training to interface with tensorflow
            def value_and_gradient(params):
                num_inputs, num_outputs = x.shape[1], y.shape[1]
                w_size = num_inputs * num_outputs
                w = params[:w_size].reshape((num_inputs, num_outputs))
                b = params[w_size:w_size + num_outputs]
                extra_vars = params[w_size + num_outputs:]
                self.sess.run(assign_ops, feed_dict=dict(zip(placeholders, [w, b, extra_vars])))
                loss, grad = self._run(train_op, X, y)
                loss = np.float64(loss)
                grad = np.float64(np.concatenate(grad, axis=None))
                return loss, grad
        
            # perform the training using scipy
            x0 = np.concatenate(self.sess.run(weights), axis=None)
            opt.minimize(value_and_gradient, x0=x0, jac=True, **self.optimizer_args)
        
        return self.evaluate_losses(X, y)
    
    def clear(self):
        TensorflowPerceptron.clear(self)
        del self.place_and_assign
