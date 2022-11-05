import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaline import AdalineGD
from helper import plot_decision_regions

# load iris data
s = '../../dataset/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# selection of setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal lenght and pedal length
X = df.iloc[0:100, [0, 2]].values

# standarize X's
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


ada1 = AdalineGD(n_iter=50, eta=0.01, randon_state=1).fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada1)
plt.title('Adaline - Stochastic Gradient descent')
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(range(1, len(ada1.losses_) +1), ada1.losses_, marker='o' )
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average loss')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=50, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) +1), ada2.losses_, marker='o' )
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Average loss')
ax[1].set_title('Adaline - Learning rate 0.0001')


plt.show()
