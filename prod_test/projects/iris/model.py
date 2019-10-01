import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def get_results(y_test, preds):
    results = dict()
    results['accuracy'] = accuracy_score(y_test, preds)
    results['classification_report'] = classification_report(y_test, preds, output_dict=True)
    results['confusion_matrix'] = confusion_matrix(y_test, preds)
    return results

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

y = pd.DataFrame(iris.target, columns=['species'])

data = pd.concat([X, y], axis=1)
data.columns = ['sepal_length', 'sepal_width', 
                'petal_length', 'petal_width', 'species']
data.species = pd.Series([iris.target_names[i] for i in data.species])
data = data.query('species in ["setosa", "virginica"]')


pred_col = 'species'
float_cols = [col for col in data.columns if col != pred_col]
X = data.drop(pred_col, axis=1)
y = data[pred_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

float_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

transformer_list = [('float', float_transformer, float_cols)]
preprocessor = ColumnTransformer(transformers=transformer_list)

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=10))])

model.fit(X_train, y_train)

preds = model.predict(X_test)

get_results(y_test, preds)
