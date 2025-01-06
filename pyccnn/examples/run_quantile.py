import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():
        
    # data
    def get_data(num=20000, std=0.2, start=0, end=4):
        x = np.sort(np.random.rand(num) * (end - start) + start)
        y = x + np.sin(np.pi * x / 2) + np.random.randn(len(x)) * std
        return x, y
        
    X, Y = get_data()
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'quantile.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_test, valid_y=y_test, **train_args)
    
    # generate predictions in interval
    min_x, max_x = np.min(X_test), np.max(X_test)
    X_range = np.linspace(min_x, max_x, 500)
    y_pred = ccnn.predict(X_range.reshape((-1, 1)))[0].flatten()
    
    # plot predicted against actual
    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X_range, y_pred, color='blue', label='10th percentile')
    ax.scatter(X_test.flatten(), y_test.flatten(), color='black', s=5, label='test data points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Quantile Regression for sin(x)')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()