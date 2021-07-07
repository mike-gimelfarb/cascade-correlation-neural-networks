import math
import tensorflow.compat.v1 as tf

from cascor._settings import dtype, epsilon


def linear(inputs):
    return inputs


def half_sigmoid(inputs):
    return tf.nn.sigmoid(inputs) - 0.5


def generalized_sigmoid(inputs):
    stats = tf.Variable([0.54132, 0.54132], dtype=dtype, name='stats')
    Q = tf.math.log(1.0 + tf.math.exp(stats[0]))
    nu = tf.math.log(1.0 + tf.math.exp(stats[1])) + epsilon
    y_pred = 1.0 / (1.0 + Q * tf.math.exp(-inputs)) ** nu
    return y_pred, stats


def swish(inputs):
    stats = tf.Variable([0.54132], dtype=dtype, name='beta')
    beta = tf.math.log(1.0 + tf.math.exp(stats[0]))
    y_pred = inputs * tf.nn.sigmoid(beta * inputs)
    return y_pred, stats


def gaussian(inputs):
    return tf.math.exp(-0.5 * inputs ** 2)

    
def generalized_gaussian(inputs):
    stats = tf.Variable([0.54132], dtype=dtype, name='power')
    power = tf.math.log(1.0 + tf.math.exp(stats[0])) + 2.0
    y_pred = tf.math.exp(-0.5 * tf.abs(inputs) ** power)
    return y_pred, stats

    
def square_rbf(inputs):
    r = tf.abs(inputs)
    y_pred = 0.5 * (2.0 - r) ** 2
    y_pred = tf.where(tf.less(r, 1.0), 1.0 - 0.5 * r ** 2, y_pred)
    y_pred = tf.where(tf.greater(r, 2.0), tf.zeros_like(r), y_pred)
    return y_pred

    
def rational_quadratic(inputs):
    stats = tf.Variable([0.54132], dtype=dtype, name='power')
    power = tf.math.log(1.0 + tf.math.exp(stats[0])) + epsilon
    y_pred = tf.math.pow(1.0 + inputs ** 2 / power, -power)
    return y_pred, stats


def periodic(inputs):
    stats = tf.Variable([0.54132], dtype=dtype, name='scale')
    scale = tf.math.log(1.0 + tf.math.exp(stats[0])) + epsilon
    y_pred = tf.math.exp(-2.0 * (scale * tf.math.sin(inputs)) ** 2)
    return y_pred, stats


def fourier(inputs):
    y_pred = 0.5 * (tf.math.cos(inputs + 1.5 * math.pi) + 1.0)
    y_pred = tf.where(tf.less(inputs, -0.5 * math.pi), tf.zeros_like(inputs), y_pred)
    y_pred = tf.where(tf.greater(inputs, 0.5 * math.pi), tf.ones_like(inputs), y_pred)
    return y_pred


def matern_3(inputs):
    d = tf.math.abs(inputs)
    return (1.0 + math.sqrt(3.0) * d) * tf.math.exp(-math.sqrt(3.0) * d)


def matern_5(inputs):
    d = tf.math.abs(inputs)
    return (1.0 + math.sqrt(5.0) * d + (5.0 / 3.0) * d ** 2) * tf.math.exp(-math.sqrt(5.0) * d)


def sinc(inputs):
    x = tf.where(tf.abs(inputs) < epsilon, epsilon * tf.ones_like(inputs), inputs)
    return tf.sin(x) / x


def ricker(inputs):
    return (1.0 - inputs ** 2) * tf.math.exp(-0.5 * inputs ** 2)


def poisson(inputs):
    return (1.0 / math.pi) * (1.0 - inputs ** 2) / (1.0 + inputs ** 2) ** 2
    
    
def morlet(inputs):
    stats = tf.Variable([0.54132], dtype=dtype, name='period')
    period = tf.math.log(1.0 + tf.math.exp(stats[0]))
    y_pred = tf.math.exp(-0.5 * inputs ** 2) * tf.math.cos(period * inputs)
    return y_pred, stats
