from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import resample
import numpy as np


class StackClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model_class:ClassifierMixin, num_estimators:int = 10, estimators:list = [], estimator_params:dict = {}) -> None:
        super().__init__()
        self.num_estimators = num_estimators
        self.model_class = model_class
        self.estimator_params = estimator_params
        self.num_estimators = num_estimators
        self.estimators = estimators


    def fit(self, x, y, sample_ratio = 0.618, **kwargs):
        if len(self.estimators) != self.num_estimators:
            self.estimators = [self.model_class(**self.estimator_params) for _ in range(self.num_estimators)]
        for i in range(self.num_estimators):
            x_tr, y_tr = resample(x, y, replace=True, n_samples=int(sample_ratio * len(x)))
            self.estimators[i].fit(x_tr, y_tr, **kwargs)
        return self


    def predict_proba(self, x):
        return np.mean([md.predict_proba(x) for md in self.estimators], axis=0)
        

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)