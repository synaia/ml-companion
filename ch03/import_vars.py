import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def get_vars():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    return (X_train, X_test, y_train, y_test, X_combined, y_combined)
