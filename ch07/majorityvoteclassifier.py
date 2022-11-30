from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

PROBA = 'probability'
CLAZZ = 'classlabel'


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in (PROBA, CLAZZ):
            raise ValueError(f"vote must be '{PROBA}'" or f'{CLAZZ}; got (vote={self.vote})')
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifier and weights must be equal; got {len(self.weights)} weights, '
                             f'{len(self.classifiers)} classifiers')
        # LabelEncoder to ensure class label start with 0; important for np.argmax
        self.labenc_ = LabelEncoder()
        self.labenc_.fit(y)
        self.classes_ = self.labenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == PROBA:
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([
                clf.predict(X) for clf in self.classifiers_
            ]).T

        maj_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=1, arr=predictions
        )
        maj_vote = self.labenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out
