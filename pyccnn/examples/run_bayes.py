import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():

    # data
    X = np.linspace(-0.5, 0.5, 100).reshape(-1, 1)
    y = 10. * np.sin(2 * np.pi * X) + np.random.randn(*X.shape)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'bayes.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_valid, valid_y=y_valid, **train_args)
    
    # generate predictions in interval
    X_test = np.linspace(-2, 2, 500).reshape((-1, 1))
    y_pred, y_pred_cov = ccnn.predict(X_test.reshape((-1, 1)))
    y_pred = y_pred.flatten()
    y_var = np.diag(y_pred_cov).flatten()
    lower_ci = y_pred - 1.96 * np.sqrt(y_var)
    upper_ci = y_pred + 1.96 * np.sqrt(y_var)
    
    # plot confidence intervals and test data
    _, ax = plt.subplots(figsize=(5, 5))
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

if __name__ == '__main__':
    run()