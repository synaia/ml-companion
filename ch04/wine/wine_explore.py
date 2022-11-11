import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from help.helper import get_wine_vars

X_train, X_test, y_train, y_test, X_train_std, X_test_std = get_wine_vars()

mms = MinMaxScaler()
# ONLY fit ON train
X_train_norm = mms.fit_transform(X_train)
# then transform ON test
X_test_norm = mms.transform(X_test)

rbs = RobustScaler()
X_train_rb = rbs.fit_transform(X_train)
X_test_rb  = rbs.transform(X_test)

# Logistic Regression impl
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


print()