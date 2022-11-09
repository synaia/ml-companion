import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ch03.import_vars import get_vars
from help.helper import plot_decision_regions

X_train, X_test, y_train, y_test, X_combined, y_combined = get_vars()
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


