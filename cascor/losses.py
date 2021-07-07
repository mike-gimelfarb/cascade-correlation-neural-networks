import tensorflow.compat.v1 as tf

from cascor._settings import dtype, epsilon
    
   
# ==================================================================================
# LOSS FUNCTIONS FOR OUTPUTS
# ==================================================================================
def negative_cross_entropy(y_true, y_pred):
    return -tf.reduce_sum(y_true * tf.math.log(y_pred + epsilon)) / tf.cast(tf.shape(y_true)[0], dtype)


def mse(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def build_pseudo_huber(delta=1.0):

    def pseudo_huber(y_true, y_pred):
        return tf.reduce_mean(delta ** 2 * (tf.math.sqrt(1.0 + ((y_true - y_pred) / delta) ** 2) - 1.0))
    
    return pseudo_huber


def build_quantile_loss(quantile=0.5):
    
    def quantile_loss(y_true, y_pred):
        error = y_pred - y_true
        return tf.reduce_mean(tf.math.maximum(quantile * error, (quantile - 1.0) * error))
    
    return quantile_loss


# ==================================================================================
# LOSS FUNCTIONS FOR CANDIDATES
#
# References:
# [1] http://www.psych.mcgill.ca/perpg/fac/shultz/personal/Recent_Publications_files/cc_tutorial_files/v3_document.htm
# [2] https://pdfs.semanticscholar.org/5a86/2a3758577c643d2a5e6ce3abd0054a0a61cc.pdf
# [3] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3477&rep=rep1&type=pdf
# ==================================================================================
def S_cascor(y_true, y_pred):
    center_y_true = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    center_y_pred = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)
    covariances = tf.reduce_mean(center_y_true * center_y_pred, axis=0)
    sum_of_squares = tf.reduce_mean(center_y_true ** 2, axis=0)
    total_covariance = tf.reduce_mean(tf.abs(covariances))
    total_sum_of_squares = tf.reduce_mean(sum_of_squares)
    return -tf.math.divide_no_nan(total_covariance, total_sum_of_squares)

    
def S1(y_true, y_pred):
    covariances = tf.reduce_mean(y_true * y_pred, axis=0)
    sum_of_squares = tf.reduce_mean(y_pred ** 2, axis=0)
    correlations = tf.math.divide_no_nan(covariances ** 2, sum_of_squares)
    return -tf.reduce_sum(correlations)


def S_mse(y_true, y_pred):
    center_y_pred = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)
    center_y_true = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    scale_y_pred = tf.math.divide_no_nan(center_y_pred, tf.math.reduce_std(y_pred, axis=0, keepdims=True))
    scale_y_true = tf.math.divide_no_nan(center_y_true, tf.math.reduce_std(y_true, axis=0, keepdims=True))
    sum_of_squares = tf.reduce_mean((scale_y_true - scale_y_pred) ** 2, axis=0)
    return tf.reduce_sum(sum_of_squares)


def fully_bayesian(y_true, y_pred):
    N = tf.cast(tf.shape(y_true)[0], dtype)
    sum_of_squares = tf.reduce_sum(y_true ** 2)
    S = -N * S1(y_true, y_pred)
    return 0.5 * tf.math.log(sum_of_squares - S + epsilon)
    
        
# ==================================================================================
# METRICS
# ==================================================================================
def accuracy(y_true, y_pred):
    prediction = tf.argmax(y_pred, 1)
    actual = tf.argmax(y_true, 1)
    return 100.0 * tf.reduce_mean(tf.cast(tf.equal(prediction, actual), dtype))


def rmse(y_true, y_pred):
    return tf.math.sqrt(mse(y_true, y_pred))


def fvu(y_true, y_pred):
    unexplained = tf.reduce_sum((y_true - y_pred) ** 2)
    centered_y_true = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    total = tf.reduce_sum(centered_y_true ** 2)
    return tf.math.divide_no_nan(unexplained, total)

