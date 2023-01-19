import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neuralnet import int_to_onehot
from neuralnet import NeuralNetMLP

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
X = ((X/255.) - .5) * 2
y = y.astype(int).values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# iterate over training epochs
for i in range(num_epochs):
    #iterate over minibatches
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break


def mse_loss(targets, probas, num_labels=10):
    onehot_target = int_to_onehot(targets, num_labels)
    return np.mean((onehot_target - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)
_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')


print() #debug here
