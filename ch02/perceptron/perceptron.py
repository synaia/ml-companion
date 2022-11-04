import numpy as np


class Perceptron:
    """Perceptron classifier

    Parameters
    ----------
    eta: float
        learning rate (btw 0. and 1.)
    n_iter: int
        passes over the training dataset.
    random_state: int
        random number generator for random weight initialization.

    Attributes
    ----------
    w_: 1d-array
        weights after fitting.
    b_: scalar
        bias unit after fitting.
    errors_: list
        number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, randon_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = randon_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ---------
        X:  {array-like}, shape = [n_samples, n_features] training vector
        y: {array-like}, shape = [n_samples]

        Returns
        -------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
