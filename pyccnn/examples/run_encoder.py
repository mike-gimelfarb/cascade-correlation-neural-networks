import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
 
from pyccnn.core import activations, losses
from pyccnn.core.model import CCNN, encoder_option
from pyccnn.core.monitor import EarlyStoppingMonitor
from pyccnn.core.units.perceptron import TensorflowPerceptron

def run():
        
    # ==================================================================================
    # DATA
    # ==================================================================================  
    # load data
    dataset = load_digits()
    n_samples = len(dataset.images)
    X = dataset.images.reshape((n_samples, -1))
    X = X / np.max(X)
    Y = X
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # ==================================================================================
    # MODEL
    # ==================================================================================  
    # layer for outputs
    output_unit = TensorflowPerceptron(activations=[tf.nn.sigmoid],
                                       loss_function=losses.mse,
                                       stopping_rule=EarlyStoppingMonitor(1e-3, 500, 5000, normalize=True),
                                       optimizer=tf.train.AdamOptimizer,
                                       optimizer_args={'learning_rate' : 0.005},
                                       batch_size=512)
    
    # layer for candidates
    candidate_unit = TensorflowPerceptron(activations=[activations.gaussian] * 5,
                                          loss_function=losses.S_cascor,
                                          stopping_rule=EarlyStoppingMonitor(1e-3, 500, 5000, normalize=True),
                                          optimizer=tf.train.AdamOptimizer,
                                          optimizer_args={'learning_rate' : 0.005},
                                          batch_size=512)
    
    # cascade correlation network
    ccnn = CCNN(64, 64,
                output_unit=output_unit,
                candidate_unit=candidate_unit,
                metric_function=losses.mae,
                lambda_param=0.9,
                output_connection_types=encoder_option)
      
    # ==================================================================================
    # TRAINING
    # ==================================================================================  
    result = ccnn.train(X_train, y_train,
                        stopping_rule=EarlyStoppingMonitor(1e-10, 18, 18),
                        valid_X=X_test, valid_y=y_test)
    
    # ==================================================================================
    # PLOTTING
    # ================================================================================== 
    # generate reconstructions
    n_test_r, n_test_c = 10, 5
    n_test = n_test_r * n_test_c
    y_true = y_test[n_test:]
    y_pred = ccnn.predict(X_test[n_test:])[0]
    
    # plot predicted against actual
    i = 0
    fig, axs = plt.subplots(n_test_r, n_test_c * 2)
    for y in range(n_test_r):
        for x in range(n_test_c):
            axs[y, x].imshow(y_true[i, :].reshape((8, 8)), cmap='Greys')
            axs[y, x + n_test_c].imshow(y_pred[i, :].reshape((8, 8)), cmap='Greys')
            axs[y, x].set_axis_off()        
            axs[y, x + n_test_c].set_axis_off()
            i += 1
    plt.show()
    

if __name__ == '__main__':
    run()