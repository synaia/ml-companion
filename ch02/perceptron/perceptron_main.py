import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from helper import plot_decision_regions

# load iris data
s = '../../dataset/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# selection of setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal lenght and pedal length
X = df.iloc[0:100, [0, 2]].values

# plot the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='o', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
# plt.show()


# train data
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.clf()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.xlabel('Number of updates')
# plt.show()

# plot decision region
plt.clf()
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()


print()