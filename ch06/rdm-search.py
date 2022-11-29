import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy import stats
from help.helper import get_wisconsin_data, get_diabetes_data

# X_train, X_test, y_train, y_test = get_wisconsin_data()
X_train, X_test, y_train, y_test = get_diabetes_data()

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
np.random.seed(1)
param_range = stats.loguniform(0.0001, 1000.0)
param_grid =[
    {'svc__C': param_range,
     'svc__kernel': ['linear']
     },
    {'svc__C': param_range,
     'svc__gamma': param_range,
     'svc__kernel': ['rbf']
     }]
gs = RandomizedSearchCV(estimator=pipe_svc,
                  param_distributions=param_grid,
                  scoring='accuracy',
                  random_state=1,
                  n_iter=20,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# Important (^ ^)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')