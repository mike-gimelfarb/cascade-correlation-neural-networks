import numpy as np
import tensorflow.compat.v1 as tf  # if using tensorflow version >= 2

from cascor.units.unit import CCNNUnit
from cascor._settings import dtype


class SklearnWrapper(CCNNUnit):
    
    def __init__(self, sklearn_model, **optional_args):
        self.sklearn_model = sklearn_model
        self.optional_args = optional_args
        
    def build(self, num_inputs, num_outputs, num_targets):
        if num_outputs != num_targets:
            raise Exception('num. outputs != num. targets: cannot use bayesian linear regression as candidate!')
        
        self.num_outputs = num_outputs
        self.models = [self.sklearn_model(**self.optional_args) for _ in num_inputs]
    
    def evaluate_losses(self, X, y):
        losses = []
        for model, x in zip(self.models, X):
            y_pred = model.predict(x)
            loss = -abs(model.score(x, y))
            losses.append(loss)
        return np.array(losses)
    
    def train(self, X, y):
        losses = []
        for i, (model, x) in enumerate(zip(self.models, X)):
            model.fit(x, y)
            loss = -abs(model.score(x, y))
            losses.append(loss)
            print('finished training {} at index {}: loss {:.8f}'.format(type(model).__name__, i, loss))
        return np.array(losses)
        
    def finalize(self, sess, inputs, head=-1):
        with sess.graph.as_default():
            y_pred = tf.py_func(self.models[head].predict, [inputs], Tout=dtype)
            y_pred = tf.reshape(y_pred, (-1, self.num_outputs))
        return y_pred,
            
    def clear(self):
        pass
