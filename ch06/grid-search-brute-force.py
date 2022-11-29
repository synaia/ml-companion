import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from help.helper import get_wisconsin_data, get_diabetes_data

X_train, X_test, y_train, y_test = get_diabetes_data()
# X_train, X_test, y_train, y_test = get_wisconsin_data()

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid =[
    {'svc__C': param_range,
     'svc__kernel': ['linear']
     },
    {'svc__C': param_range,
     'svc__gamma': param_range,
     'svc__kernel': ['rbf']
     }]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  refit=True, # get best_estimator_ param
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# Important (^ ^)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')
