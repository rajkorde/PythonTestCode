from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


class RandomBinaryClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        self.table = y.value_counts(normalize=True).reset_index()
        self.table.columns = ['Value', 'Count']

        return self

    def predict(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        return pd.Series(np.random.choice(self.table.Value.values,
                         size=len(X),
                         p=self.table.Count.values))

    def score(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        pred = self.predict(X)
        return accuracy_score(y, pred)
