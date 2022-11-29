import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import SVC
from scipy import stats
from help.helper import get_wisconsin_data

X_train, X_test, y_train, y_test = get_wisconsin_data()

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
hs = HalvingRandomSearchCV(estimator=pipe_svc,
                           param_distributions=param_grid,
                           n_candidates='exhaust',
                           resource='n_samples',
                           factor=1.5,
                           random_state=1,
                           n_jobs=-1)


hs = hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)

# Important (^ ^)
clf = hs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')