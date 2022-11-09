import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from help.helper import plot_decision_regions
from ch03.import_vars import get_vars

X_train, X_test, y_train, y_test, X_combined, y_combined = get_vars()

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)

feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()


plot_decision_regions(X_combined, y_combined, classifier=tree_model, test_idx=range(105, 150))
plt.xlabel('Sepal length [standarized]')
plt.ylabel('Petal length [standarized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
