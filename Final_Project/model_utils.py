"""
Shared utility classes for the Telco Customer Churn project.

This module must be importable by BOTH the training notebook and app.py,
since the fitted pipeline saved to saved_model.pkl references the
ColumnSelector class defined here.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Lightweight transformer that selects a fixed subset of columns from a
    (dense or sparse) numeric array, based on a boolean mask determined
    once during feature selection.

    This avoids pickling a full feature-selection estimator (e.g. a large
    Random Forest used only to rank feature importance) inside the final
    production model artifact — only the boolean mask is stored.
    """

    def __init__(self, mask=None):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        return X[:, self.mask]

    def get_support(self):
        return self.mask
