import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sbs import SBS
from help.helper import get_wine_vars

X_train, X_test, y_train, y_test, X_train_std, X_test_std = get_wine_vars()
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim(0.7, 1.02)
plt.ylabel('Accuracy')
plt.xlabel('number of features')
plt.grid()
plt.tight_layout()
plt.show()

from sklearn.feature_selection import SequentialFeatureSelector