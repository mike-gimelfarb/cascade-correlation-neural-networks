import time
import warnings
import itertools
import numpy as np
import tensorflow.compat.v1 as tf  # if using tensorflow version >= 2

from cascor.monitor import LossHistoryMonitor
from cascor._settings import dtype


# ==================================================================================
# CCNN CONNECTION TYPES
# ==================================================================================
def siblings(features, num_inputs, num_hidden):
    if len(num_hidden) == 0:
        return [(features, True)]
    else:
        return [(features[:,:-num_hidden[-1]], False)]


def descendants(features, num_inputs, num_hidden):
    return [(features, True)]


def siblings_descendants(features, num_inputs, num_hidden):
    return siblings(features, num_inputs, num_hidden) + descendants(features, num_inputs, num_hidden)


def encoder_option(features, num_inputs, num_hidden):
    if len(num_hidden) == 0:
        return [(tf.ones_like(features[:,:1]), True)]
    else:
        return [(features[:, num_inputs:], True)]


# ==================================================================================
# PLAIN CCNN
# ==================================================================================
class CCNN:

    def __init__(self, num_inputs, num_outputs, output_unit, candidate_unit, metric_function, lambda_param=1.0,
                 output_connection_types=descendants, candidate_connection_types=siblings_descendants):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_unit = output_unit
        self.candidate_unit = candidate_unit
        self.metric_function = metric_function
        self.lambda_param = lambda_param
        self.output_connection_types = output_connection_types
        self.candidate_connection_types = candidate_connection_types

        # data
        self.num_hidden = []
        self.loss_data = LossHistoryMonitor()
        
        # graph
        tf.disable_eager_execution()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph) 
        with self.graph.as_default():
            self.inputs = tf.placeholder(dtype=dtype, shape=[None, num_inputs], name='inputs')
            self.features = self.inputs
            self.targets = tf.placeholder(dtype=dtype, shape=[None, num_outputs], name='targets')
            self.preds = tf.placeholder(dtype=dtype, shape=[None, num_outputs], name='predictions')
            self.metric = metric_function(self.targets, self.preds)
    
    def close(self):
        self.sess.close()
        del self.inputs, self.features, self.targets, self.preds, self.metric, self.graph
        del self.num_hidden, self.loss_data        
        
    # ==================================================================================
    # PREDICTION OPERATIONS
    # ==================================================================================    
    def predict(self, X):
        X = X.reshape((-1, self.num_inputs))
        return self.sess.run(self.y_pred, feed_dict={self.inputs: X})
    
    def predict_features(self, X):
        X = X.reshape((-1, self.num_inputs))
        return self.sess.run(self.features, feed_dict={self.inputs: X})
    
    def topology(self):
        return [self.num_inputs] + self.num_hidden + [self.num_outputs]
        
    # ==================================================================================
    # TRAINING OPERATIONS
    # ==================================================================================
    def train_outputs(self, train_X, train_y, valid_X, valid_y):
        
        # output connection types
        connection_types = self.output_connection_types(self.features, self.num_inputs, self.num_hidden)
        num_inputs = [inputs.get_shape()[1] for inputs, _ in connection_types]
        all_inputs = [inputs for inputs, _ in connection_types]
        
        # build output neurons
        self.output_unit.build(num_inputs, self.num_outputs, self.num_outputs)
        
        # train output neurons
        print('training output neurons...')
        train_features = self.sess.run(all_inputs, feed_dict={self.inputs: train_X})
        train_losses = self.output_unit.train(train_features, train_y)
        
        # pick the best neuron and add it to the graph
        best_index = self._add_best_output(train_losses, connection_types)
        
        # evaluate losses
        valid_features = self.sess.run(all_inputs, feed_dict={self.inputs: valid_X})
        valid_losses = self.output_unit.evaluate_losses(valid_features, valid_y)
        self.loss_data.append(loss=train_losses[best_index], valid_loss=valid_losses[best_index])
        
        # evaluate metrics
        train_y_pred = self.sess.run(self.y_pred[0], feed_dict={self.inputs: train_X})
        valid_y_pred = self.sess.run(self.y_pred[0], feed_dict={self.inputs: valid_X})
        train_metric = self.sess.run(self.metric, feed_dict={self.targets: train_y, self.preds: train_y_pred})
        valid_metric = self.sess.run(self.metric, feed_dict={self.targets: valid_y, self.preds: valid_y_pred})
        self.loss_data.append(metric=train_metric, valid_metric=valid_metric)
        print(self.loss_data.summary())
        
        # finish        
        self.output_unit.clear()
        return train_losses[best_index], valid_losses[best_index]
    
    def _add_best_output(self, losses, connection_types):
        
        # choose the best output neuron and connection type
        best_index = np.argmin(losses)
        outputs_per_connection_type = losses.size // len(connection_types)
        best_type = best_index // outputs_per_connection_type
        inputs, _ = connection_types[best_type]
        
        # add the candidate neuron to the main graph
        self.y_pred = self.output_unit.finalize(self.sess, inputs, best_index)
        print('added output {} with input shape {}\n'.format(best_index, inputs.get_shape()))    
        return best_index    
        
    def train_candidates(self, train_X, train_y):
        
        # candidate connection types
        connection_types = self.candidate_connection_types(self.features, self.num_inputs, self.num_hidden)
        num_inputs = [inputs.get_shape()[1] for inputs, _ in connection_types]
        all_inputs = [inputs for inputs, _ in connection_types]
        
        # build candidate neurons
        self.candidate_unit.build(num_inputs, 1, self.num_outputs)
        
        # train candidate neurons
        print('training candidate neurons...')
        train_features, y_pred = self.sess.run((all_inputs, self.y_pred[0]), feed_dict={self.inputs: train_X})
        residuals = train_y - y_pred
        correlations = self.candidate_unit.train(train_features, residuals)
        
        # pick the best neuron and add it to the graph
        self._add_best_candidate(correlations, connection_types)
        self.candidate_unit.clear()

    def _add_best_candidate(self, correlations, connection_types):
        
        # identify which candidates will be added to new layers
        candidates_per_connection_type = correlations.size // len(connection_types)
        assert candidates_per_connection_type * len(connection_types) == correlations.size
        new_layer_mask = np.repeat([add_to_new_layer for _, add_to_new_layer in connection_types],
                                   candidates_per_connection_type)
        
        # give a slight advantage to candidates added to an existing layer
        if np.min(correlations) >= 0:
            advantage = (1.0 - self.lambda_param) * new_layer_mask.astype(float) + self.lambda_param
        elif np.max(correlations) <= 0:
            advantage = (1.0 - self.lambda_param) * (~new_layer_mask).astype(float) + self.lambda_param
        else:
            warnings.warn('Detected mixed signs of correlations - cannot penalize descendants!')
            advantage = np.ones_like(correlations)
        correlations = correlations * advantage
        
        # choose the best candidate neuron and connection type
        best_index = np.argmin(correlations)
        best_type = best_index // candidates_per_connection_type
        inputs, add_to_new_layer = connection_types[best_type]
        
        # add the candidate neuron to the main graph
        y_pred = self.candidate_unit.finalize(self.sess, inputs, best_index)[0]
        self.features = tf.concat([self.features, y_pred], axis=1)
        if add_to_new_layer:
            self.num_hidden.append(1)
        else:
            self.num_hidden[-1] += 1
        print('added candidate {} with input shape {}\n'.format(best_index, inputs.get_shape()))
        print('network shape {}\n'.format('-'.join(map(str, self.topology()))))
        return best_index
            
    def train(self, train_X, train_y, stopping_rule, valid_X, valid_y):
        
        # preparation
        self.loss_data.clear()
        stopping_rule.initialize()
        start_time = time.time()
        
        # train outputs
        print('========== Iteration 0 [Training Initial Network] ==========\n')
        loss, valid_loss = self.train_outputs(train_X, train_y, valid_X, valid_y)
        if stopping_rule.update(valid_loss):
            return self.loss_data
        
        # main training loop
        for t in itertools.count(1):
            print('========== Iteration {} [Network Growth] ==========\n'.format(t))
            
            # train candidates
            self.train_candidates(train_X, train_y)
            
            # train outputs
            loss, valid_loss = self.train_outputs(train_X, train_y, valid_X, valid_y)
            if stopping_rule.update(valid_loss): 
                break
        
        # finalize
        end_time = (time.time() - start_time) / float(t) 
        print('training complete, time/neuron {:.6f} s'.format(end_time))  
        return self.loss_data
    
