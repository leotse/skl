import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


from data.loaders import load_training_data, load_validation_data

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
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_val)

for p, a in zip(predictions, y_val):
    print(f'{p == a}\t{(p, a)}')
print(f'accuracy: {np.mean(predictions == y_val)}\n')
print(metrics.classification_report(y_val, predictions))
print(metrics.confusion_matrix(y_val, predictions))
