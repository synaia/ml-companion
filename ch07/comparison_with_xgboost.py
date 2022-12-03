from help.helper import get_wine_boosting_vars
from sklearn.metrics import accuracy_score
import xgboost as xgb

X_train, X_test, y_train, y_test = get_wine_boosting_vars()

model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=1, use_label_encoder=False)
gbm = model.fit(X_train, y_train)
y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)

gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)

print(f'XGboost train/tesst accuracies {gbm_train:0.3f}/{gbm_test:0.3f}')

