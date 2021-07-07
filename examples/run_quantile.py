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
from core.units.perceptron import TensorflowPerceptron, ScipyPerceptron

def run():
        
    # ==================================================================================
    # DATA
    # ==================================================================================  
    def f_predictable(x):
        return x + np.sin(np.pi * x / 2)
    
    
    def f(x, std=0.2):
        return f_predictable(x) + np.random.randn(len(x)) * std
    
    
    def get_data(num, start=0, end=4):
        x = np.sort(np.random.rand(num) * (end - start) + start)
        y = f(x)
        return x.reshape(-1, 1), y
    
    
    X, Y = get_data(num=20000)
    Y = Y.reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
    
    # ==================================================================================
    # MODEL
    # ==================================================================================  
    # layer for outputs
    output_unit = ScipyPerceptron(activations=[activations.linear], loss_function=losses.build_quantile_loss(0.1))
    
    # layer for candidates
    candidate_unit = TensorflowPerceptron(activations=[activations.gaussian] * 5,
                                          loss_function=losses.S_cascor,
                                          stopping_rule=EarlyStoppingMonitor(1e-3, 500, 10000, normalize=True),
                                          optimizer=tf.train.AdamOptimizer,
                                          optimizer_args={'learning_rate' : 0.01},
                                          batch_size=999999)
    
    # cascade correlation network
    ccnn = CCNN(1, 1,
                output_unit=output_unit, candidate_unit=candidate_unit,
                metric_function=losses.build_quantile_loss(0.05),
                lambda_param=0.9)
      
    # ==================================================================================
    # TRAINING
    # ==================================================================================  
    result = ccnn.train(X_train, y_train,
                        stopping_rule=EarlyStoppingMonitor(1e-10, 10, 10),
                        valid_X=X_test, valid_y=y_test)
    
    # ==================================================================================
    # PLOTTING
    # ================================================================================== 
    # generate predictions in interval
    min_x, max_x = np.min(X_test), np.max(X_test)
    X_range = np.linspace(min_x, max_x, 500)
    y_pred = ccnn.predict(X_range.reshape((-1, 1)))[0].flatten()
    
    # plot predicted against actual
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X_range, y_pred, color='blue', label='10th percentile')
    ax.scatter(X_test.flatten(), y_test.flatten(), color='black', s=5, label='test data points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Quantile Regression for sin(x)')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
