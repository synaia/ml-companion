import numpy as np


class AdalineGD:
    """ADAptative LInear NEuron classifier

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
    losses_: list
        mean squared error loss function values in each epoch.

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
        y: {array-like}, shape = [n_samples], target values

        Returns
        -------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X) # linear relat.
            output = self.activation(net_input)
            errors = (y - output) # knows as 'residual'
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

