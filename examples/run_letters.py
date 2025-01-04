import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import string
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
from cascor import activations, losses
from cascor.model import CCNN
from cascor.monitor import EarlyStoppingMonitor
from cascor.units.perceptron import TensorflowPerceptron


def run():
        
    # ==================================================================================
    # DATA
    # ================================================================================== 
    # load data
    num = 20000
    dataset = pd.read_csv(pathlib.Path(__file__).parent / 'data' / 'letters.csv')
    X = dataset.iloc[:num,:-1].values
    y = dataset.iloc[:num, -1].values
    
    # perform dummy encoding
    y1 = LabelEncoder().fit_transform(y)
    Y = pd.get_dummies(y1).values
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # ==================================================================================
    # MODEL
    # ==================================================================================  
    # layer for outputs
    output_unit = TensorflowPerceptron(activations=[tf.nn.softmax],
                                       loss_function=losses.negative_cross_entropy,
                                       stopping_rule=EarlyStoppingMonitor(1e-2, 50, 5000, normalize=True),
                                       optimizer=tf.train.AdamOptimizer,
                                       optimizer_args={'learning_rate': 0.005},
                                       batch_size=512)
    
    # layer for candidates
    candidate_unit = TensorflowPerceptron(activations=[activations.gaussian] * 5,
                                          loss_function=losses.S_cascor,
                                          stopping_rule=EarlyStoppingMonitor(1e-2, 50, 5000, normalize=True),
                                          optimizer=tf.train.AdamOptimizer,
                                          optimizer_args={'learning_rate': 0.005},
                                          batch_size=512)
    
    # cascade correlation network
    ccnn = CCNN(16, 26,
                output_unit=output_unit, candidate_unit=candidate_unit,
                metric_function=losses.accuracy,
                lambda_param=0.8)
    
    # ==================================================================================
    # TRAINING
    # ==================================================================================  
    result = ccnn.train(X_train, y_train,
                        stopping_rule=EarlyStoppingMonitor(1e-10, 20, 20),
                        valid_X=X_test, valid_y=y_test)
    
    # ==================================================================================
    # PLOTTING
    # ==================================================================================  
    # build confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(ccnn.predict(X_test)[0], axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(26))
    
    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')
    ax.xaxis.set_ticklabels(list(string.ascii_lowercase))
    ax.yaxis.set_ticklabels(list(string.ascii_lowercase))
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix for Letter Classification')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()