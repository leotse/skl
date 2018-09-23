import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import print_report


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key, wrap=False):
        self.key = key
        self.wrap = wrap

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        series = df.loc[:, self.key]
        if self.wrap:
            return pd.DataFrame(series)
        return series


# load training and validation data
training = pd.read_csv('data/training_data.csv')
validation = pd.read_csv('data/validation_data.csv')

X_train = training.loc[:, ['expense description', 'pre-tax amount']]
y_train = training.loc[:, ['category']].values.ravel()

X_val = validation.loc[:, ['expense description', 'pre-tax amount']]
y_val = validation.loc[:, ['category']].values.ravel()

# build data pipeline!
pipeline = Pipeline([
    ('features', FeatureUnion([

        # expense description feature
        ('description', Pipeline([
            ('selector', FeatureSelector('expense description')),
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),

        # pretax amount feature
        ('pretax', Pipeline([
            ('selector', FeatureSelector('pre-tax amount', wrap=True)),
            ('scaler', StandardScaler()),
        ]))
    ], transformer_weights={
        'description': 1.0,
        'pretax': 1.0,
    })),

    # we now have preprocessed description and pretax features
    # apply classifier
    # ('clf', DecisionTreeClassifier()),
    # ('clf', RandomForestClassifier(n_estimators=100)),
    ('clf', SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-3,
        max_iter=5,
        tol=None,
        random_state=42,
    ))
])

# gridsearch loss func and penlaty
# pipeline = GridSearchCV(pipeline, {
#     'clf__loss': ['hinge', 'log', 'perceptron'],
#     'clf__penalty': ['l2', 'l1'],
#     'clf__alpha': [1e-2, 1e-3, 1e-4, 1e-5],
#     'clf__max_iter': range(5, 21, 5),
# }, n_jobs=-1)

clf = pipeline.fit(X_train, y_train)
# print(f'best score:  {clf.best_score_}')
# print(f'best params: {clf.best_params_}')

# training data accuracy
print('=== training error ===')
predictions = clf.predict(X_train)
print_report(predictions, y_train)

# validation data accuracy
print('=== validation error ===')
predictions = clf.predict(X_val)
print_report(predictions, y_val)

# classification report
print(classification_report(y_val, predictions))
