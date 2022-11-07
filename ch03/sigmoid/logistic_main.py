from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from logistic import LogisticRessionGD
from help.helper import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRessionGD(eta=0.3, n_iter=1000, randon_state=1)

lrgd.fit(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.show()


lr = LogisticRegression(C=100., solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.show()

pr = lr.predict_proba(X_test_std[:3, :])
print(pr)

# The Inverse Regularization Parameter (C)
weights, params = [], []
for c in np.arange(-5, 5):
    inverse_reg = 10.0**c
    lr = LogisticRegression(C=inverse_reg, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(inverse_reg)
weights = np.array(weights)
plt.clf()
plt.plot(params, weights[:, 0], label='Petal length')
plt.plot(params, weights[:, 1], label='Petal width', linestyle='--')
plt.ylabel('weight coeffiencient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
