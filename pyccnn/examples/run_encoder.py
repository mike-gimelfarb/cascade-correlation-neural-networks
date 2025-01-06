import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
 
from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():
        
    # data
    dataset = load_digits()
    X = dataset.images.reshape((len(dataset.images), -1))
    X = X / np.max(X)
    X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.3, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'encoder.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_test, valid_y=y_test, **train_args)
        
    # generate reconstructions
    n_test_r, n_test_c = 10, 5
    n_test = n_test_r * n_test_c
    y_true = y_test[n_test:]
    y_pred = ccnn.predict(X_test[n_test:])[0]
    
    # plot predicted against actual
    i = 0
    _, axs = plt.subplots(n_test_r, n_test_c * 2)
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