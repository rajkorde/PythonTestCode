import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, X, y, colnames):
        self.data = pd.concat([X, y], axis=1)
        self.data.columns = colnames

    def prep(self, pred_col, float_cols, query=None):
        if query is not None:
            self.data = self.data.query(query)

        self.X = self.data[float_cols]
        self.y = self.data[pred_col]

    def split_train_test(self, test_size=0.2, random_state=1):
        return train_test_split(
            self.X, self.y, test_size=0.2, random_state=1)
