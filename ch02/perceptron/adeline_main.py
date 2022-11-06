import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from adaline import AdalineGD
from sklearn.metrics import f1_score
from help.helper import plot_decision_regions

# load iris data
# s = '../../dataset/iris.data'
# df = pd.read_csv(s, header=None, encoding='utf-8')

df = pd.read_csv('../../dataset/food-world-cup-data.csv', header=None, encoding='latin')
for c in df.iloc[:, 2:43].columns:
    df[c] = df[c].astype('category').cat.codes

# selection of setosa and versicolor
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', 0, 1)
y = df.iloc[:, 43].values
y = np.where(y == 'Female', 0, 1)

# extract sepal lenght and pedal length
# X = df.iloc[0:100, [0, 2]].values
X = df.iloc[:, 2:43].values

# standarize X's
X_std = np.copy(X)
for i in range(X.shape[1]):
    X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

# X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


ada1 = AdalineGD(n_iter=100, eta=0.01, randon_state=1).fit(X_std, y)

# plot_decision_regions(X_std, y, classifier=ada1)
# plt.title('Adaline - Stochastic Gradient descent')
# plt.xlabel('Sepal length [standarized]')
# plt.ylabel('Petal length [standarized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# Predict
r = random.randint(0, len(y))
X_test, y_test = X_std[r], y[r]
y_hat = ada1.predict(X_test)
# print(f1_score(y_test, y_hat))


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(range(1, len(ada1.losses_) +1), ada1.losses_, marker='o' )
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average loss')
ax[0].set_title('Adaline - Learning rate 0.01 / 100 iters')

ada2 = AdalineGD(n_iter=100, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) +1), ada2.losses_, marker='o' )
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Average loss')
ax[1].set_title('Adaline - Learning rate 0.0001 / 100 iters')


plt.show()
