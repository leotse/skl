import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from data.loaders import load_training_data, load_validation_data
from utils import print_results

# load training and validation data
X_train, y_train = load_training_data()
X_val, y_val = load_validation_data()

# build prediction pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-3,
        random_state=42,
        max_iter=5,
        tol=None,
    ))
])

# use grid search to find optimal params for the model
gs_clf = GridSearchCV(text_clf, {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}, n_jobs=-1)
gs_clf.fit(X_train, y_train)

# predict!
predictions = gs_clf.predict(X_val)
print_results(predictions, y_val)
print(f'best score:  {gs_clf.best_score_}')
print(f'best params: {gs_clf.best_params_}')
