import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from help.helper import get_wine_vars

X_train, X_test, y_train, y_test, X_train_std, X_test_std = get_wine_vars()

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red','cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen','lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
    Inv=10.0**c
    lr = LogisticRegression(penalty='l1', C=Inv, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(Inv)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()