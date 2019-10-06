from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FeatureSquarer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureSquarer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        for col in X.columns:
            if X[col].dtype == 'int64':
                X[col] = np.power(X[col], 2)
        return X


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Extracts columns of a certain data type from a given dataframe

    Parameters
    ----------
    dtype: Column data types to extract
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        """
        Extracts columns of a certain datatypes from a pandas dataframe

        Parameters
        ----------
        X: Input pandas dataframe
        y: Ignored
        """
        return self

    def transform(self, X):
        """
        Extracts columns of a certain datatypes from a pandas dataframe

        Parameters
        ----------
        X: Input pandas dataframe

        Returns
        -------
        A pandas dataframe with columns of given data types extracted
        """
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
