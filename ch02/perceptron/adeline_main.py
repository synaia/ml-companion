import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaline import AdalineGD

# load iris data
s = '../../dataset/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# selection of setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal lenght and pedal length
X = df.iloc[0:100, [0, 2]].values


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) +1), np.log10(ada1.losses_), marker='o' )
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) +1), np.log10(ada2.losses_), marker='o' )
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
