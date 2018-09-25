from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import FeatureSelector, IsWeekendTransformer, PipelineLabelBinarizer


# load training and validation data
employee = pd.read_csv('data/employee.csv')

training = pd.read_csv('data/training_data.csv', parse_dates=['date'])
X_train = training.merge(employee, on='employee id').loc[:, [
    'date',
    'category',
    'pre-tax amount',
    'role',
]]

test = pd.read_csv('data/validation_data.csv', parse_dates=['date'])
X_test = test.merge(employee, on='employee id').loc[:, [
    'date',
    'category',
    'pre-tax amount',
    'role',
]]

model = Pipeline([
    ('features', FeatureUnion([

        # weekend?
        ('weekend', Pipeline([
            ('selector', FeatureSelector('date')),
            ('transform', IsWeekendTransformer()),
        ])),

        # category
        ('category', Pipeline([
            ('selector', FeatureSelector('category')),
            ('encode', PipelineLabelBinarizer()),
        ])),

        # role
        ('role', Pipeline([
            ('selector', FeatureSelector('role')),
            ('encode', PipelineLabelBinarizer()),
        ])),

        # pretax amount
        ('pretax', Pipeline([
            ('selector', FeatureSelector('pre-tax amount', wrap=True)),
            ('scale', StandardScaler()),
        ])),
    ])),

    # clustering
    ('cluster', KMeans(n_clusters=2)),
])

model.fit(X_train)

# output results
print(X_train)
print(model.named_steps['cluster'].labels_)

print(X_test)
print(model.predict(X_test))
