import numpy as np


class LogisticRessionGD:
    """GD based logistic regression classifier

    Parameters
    ----------
    eta: float
        learning rate (btw 0. and 1.)
    n_iter: int
        passes over the training dataset.
    shuffle: bool (default : True)
        shuffles training data every epoch if True.
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
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))) / X.shape[0]
            self.losses_.append(loss)

        return self

    def _initialize_weights(self, m):
        """Initialize weights to small random mnumers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def net_input(self, X):
        """Calculate the net input (dot product)"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute linear activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
