import numpy as np


class AdalineGD:
    """ADAptative LInear NEuron classifier

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
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, randon_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
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
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random mnumers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adeline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)   # residuals
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss


        #     net_input = self.net_input(X) # linear relat.
        #     output = self.activation(net_input)
        #     errors = (y - output) # knows as 'residual'
        #     self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
        #     self.b_ += self.eta * 2.0 * errors.mean()
        #     loss = (errors**2).mean()
        #     self.losses_.append(loss)
        # return self
    #
    # def _initialize_weights(self, m):
    #     """Initialize weights to small random numbers"""
    #     self.rgen = np.random.RandomState(self.random_state)
    #     self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
    #     self.b_ = np.float_(0.0)
    #     self.w_initialized = True

    def net_input(self, X):
        """Calculate the net input (dot product)"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
        # return np.where(self.activation(self.net_input(X)))

