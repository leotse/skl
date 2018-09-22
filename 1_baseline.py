import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from data.loaders import load_training_data, load_validation_data
from utils import print_report

# load training and validation data
X_train, y_train = load_training_data()
X_val, y_val = load_validation_data()

# build token matrix
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# apply tf-idf transformation
# skimming the training data however, tf counts are all 0..
# so we are technically only applying idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# use naive bayes classifier as the baseline
X_val_counts = count_vect.transform(X_val)
X_val_tfidf = tfidf_transformer.transform(X_val_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# training error
print('=== training error ===')
predictions = clf.predict(X_train_tfidf)
print_report(predictions, y_train)

# validation error
print('=== validation error ===')
predictions = clf.predict(X_val_tfidf)
print_report(predictions, y_val)
