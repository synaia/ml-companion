import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from ch03.import_vars import get_vars
from help.helper import plot_decision_regions

X_train, X_test, y_train, y_test, X_combined, y_combined = get_vars()
# p=1 Manhattan distance and p=2  Euclidean distance, as generalization of Minkowski
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
