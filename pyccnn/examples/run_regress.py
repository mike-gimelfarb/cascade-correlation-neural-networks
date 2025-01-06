import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():
        
    # data        
    X = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
    y = 10. * np.sin(2 * np.pi * X) + np.random.randn(*X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'regress.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_test, valid_y=y_test, **train_args)
    
    # generate predictions in interval
    min_x, max_x = np.min(X_test), np.max(X_test)
    X_range = np.linspace(min_x, max_x, 500)
    y_pred = ccnn.predict(X_range.reshape((-1, 1)))[0].flatten()
    
    # plot predicted against actual
    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X_range, y_pred, color='blue', label='prediction')
    ax.scatter(X_test.flatten(), y_test.flatten(), color='black', s=5, label='test data points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Regression for sin(x)')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()