import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from core.utils.common_transformers import TypeSelector, FeatureSquarer

sys.path.append('D:/GitRepos/github/PythonTestCode/prod_test')

df = pd.DataFrame(data=[[1, 2, 'chad'], [4, 5, 'John']],
    columns=['col1', 'col2', 'col3'])
    
float_pipeline = Pipeline(steps=[('float_squarer', FeatureSquarer())])
float_pipeline.fit(df)
float_pipeline.transform(df)


transformer_list = [('float', float_pipeline, ['col1', 'col2'])]
preprocessor = ColumnTransformer(transformer_list)

int_data = Pipeline(steps=[('column_extractor', TypeSelector('int64'))])