import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from help.helper import get_wine_vars, plot_decision_regions

X_train, X_test, y_train, y_test, X_train_std, X_test_std = get_wine_vars()
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca  = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()



columnslabel = ['Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity od ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyamins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# plt.clf()
fig, ax = plt.subplots()
ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loading for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(columnslabel, rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()