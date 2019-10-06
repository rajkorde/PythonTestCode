import sys

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from core.utils.common_transformers import TypeSelector, FeatureSquarer
from core.utils.common_estimators import RandomBinaryClassifier

# sys.path.append('D:/GitRepos/github/PythonTestCode/prod_test')


# transformer tests
df = pd.DataFrame(data=[[1, 2, 'chad'], [4, 5, 'John']],
                  columns=['col1', 'col2', 'col3'])

float_pipeline = Pipeline(steps=[('float_squarer', FeatureSquarer())])
float_pipeline.fit(df)
float_pipeline.transform(df)

transformer_list = [('float', float_pipeline, ['col1', 'col2'])]
preprocessor = ColumnTransformer(transformer_list)

int_data = Pipeline(steps=[('column_extractor', TypeSelector('int64'))])

int_data.fit_transform(df)

# estimator test
X = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
y = pd.Series(np.random.choice(['setosa', 'virginica'], 100, p=[0.3, 0.7]))
y_test = pd.Series(np.random.choice(['setosa', 'virginica'],
                                    100, p=[0.3, 0.7]))

model = RandomBinaryClassifier()

model.fit(X, y)

y = model.predict(X)
y.value_counts()

model.score(X, y_test)
