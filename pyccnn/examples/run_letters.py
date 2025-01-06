import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import string
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
from pyccnn.core.model import CCNN
from pyccnn.core.parser import load_config

def run():
        
    # data
    num = 20000
    dataset = pd.read_csv(pathlib.Path(__file__).parent / 'data' / 'letters.csv')
    X = dataset.iloc[:num,:-1].values
    y = dataset.iloc[:num, -1].values
    y1 = LabelEncoder().fit_transform(y)
    Y = pd.get_dummies(y1).values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'letters.ini')
    network_args, train_args = load_config(path)
    ccnn = CCNN(**network_args)
    ccnn.train(X_train, y_train, valid_X=X_test, valid_y=y_test, **train_args)
    
    # build confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(ccnn.predict(X_test)[0], axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(26))
    
    # plot the confusion matrix
    _, ax = plt.subplots(figsize=(10, 8))
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