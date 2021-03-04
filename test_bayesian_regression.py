import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
 
from core import activations, losses
from core.model import CCNN
from core.monitor import EarlyStoppingMonitor
from core.units.linear import BayesianLinear 
from core.units.perceptron import TensorflowPerceptron


# ==================================================================================
# DATA
# ==================================================================================  
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10. * np.sin(2 * np.pi * (x)) + epsilon


train_size = 200
noise = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

# ==================================================================================
# MODEL
# ==================================================================================  
# layer for outputs
output_unit = BayesianLinear(alpha=0.1, beta=5.0)

# layer for candidates
candidate_unit = TensorflowPerceptron(activations=[activations.gaussian] * 5,
                                      loss_function=losses.fully_bayesian,
                                      stopping_rule=EarlyStoppingMonitor(1e-5, 500, 10000, normalize=True),
                                      optimizer=tf.train.AdamOptimizer,
                                      optimizer_args={'learning_rate' : 0.01},
                                      batch_size=999999,
                                      regularizer=tf.nn.l2_loss,
                                      reg_penalty=1e-6)

# cascade correlation network
ccnn = CCNN(1, 1,
            output_unit=output_unit, candidate_unit=candidate_unit,
            metric_function=losses.fvu,
            lambda_param=0.9)

# ==================================================================================
# TRAINING
# ==================================================================================  
result = ccnn.train(X_train, y_train,
                    stopping_rule=EarlyStoppingMonitor(1e-5, 2, 20),
                    valid_X=X_valid, valid_y=y_valid)

# ==================================================================================
# PLOTTING
# ==================================================================================      
# generate predictions in interval
X_test = np.linspace(-2, 2, 500).reshape((-1, 1))
y_pred, y_pred_cov = ccnn.predict(X_test.reshape((-1, 1)))
y_pred = y_pred.flatten()
y_var = np.diag(y_pred_cov).flatten()
lower_ci = y_pred - 1.96 * np.sqrt(y_var)
upper_ci = y_pred + 1.96 * np.sqrt(y_var)

# plot confidence intervals and test data
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_test, y_pred, color='blue', label='mean prediction')
ax.plot(X_test, lower_ci, color='gray')
ax.plot(X_test, upper_ci, color='gray')
ax.fill_between(X_test.flatten(), lower_ci, upper_ci, alpha=0.1, color='gray', label='95% credible region')
ax.scatter(X_train.flatten(), y_train.flatten(), color='black', s=5, label='train data points')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Bayesian Linear Regression for sin(x)')
ax.legend(loc='best')
plt.tight_layout()
plt.show()
