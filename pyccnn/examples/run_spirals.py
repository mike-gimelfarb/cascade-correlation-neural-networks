import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():
        
    # data    
    def generate_two_spirals(n_points, noise=.5):
        n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
    
    X, y = generate_two_spirals(1000)
    plt.title('training set')
    plt.plot(X[y == 0, 0], X[y == 0, 1], '.', label='class 1')
    plt.plot(X[y == 1, 0], X[y == 1, 1], '.', label='class 2')
    plt.legend()
    plt.show()
    y1 = LabelEncoder().fit_transform(y)
    Y = pd.get_dummies(y1).values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'spirals.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_test, valid_y=y_test, **train_args)    
    
    # plot the boundary for classification
    def plot_boundary(xx, yy, samples, classes, probs): 
        f, ax = plt.subplots(figsize=(5, 5))
        contour = ax.contourf(xx, yy, probs, 100, cmap="RdBu", vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])
        K = np.unique(classes).size
        for k in range(K):
            xyk = samples[classes == k]
            ax.scatter(xyk[:, 0], xyk[:, 1], s=6, label='class {}'.format(k))
        plt.legend()
        plt.show()
        
    samples, classes = generate_two_spirals(1000)
    xmin, xmax = -15., 15.
    ymin, ymax = -15., 15.
    xstp, ystp = 0.05, 0.05
    xx, yy = np.mgrid[xmin:xmax:xstp, ymin:ymax:ystp]
    grid = np.c_[xx.ravel(), yy.ravel()]
    y_pred = ccnn.predict(grid)[0][:, 0].reshape(xx.shape)
    plot_boundary(xx, yy, samples, classes, y_pred)

if __name__ == '__main__':
    run()