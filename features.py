import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer


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


class IsWeekendTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, series):
        weekdays = series.map(lambda date: date.weekday())
        return pd.DataFrame(weekdays >= 5).astype(int)


class PipelineLabelBinarizer(LabelBinarizer):
    def __init__(self):
        super().__init__()

    def fit_transform(self, x, y=None):
        return super().fit_transform(x)
