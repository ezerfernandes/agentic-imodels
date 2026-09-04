"""A small, human-readable decision-tree regressor."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

from ._names import (
    align_feature_names,
    format_feature_name,
    input_feature_names,
    resolve_feature_names,
    validate_fit_data,
    validate_predict_data,
)


class TinyDTDepth2Regressor(RegressorMixin, BaseEstimator):
    """Tiny decision tree regressor with a maximum depth of two."""

    def __init__(self, max_depth=2, min_samples_leaf=4, random_state=42, feature_names=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_names = feature_names

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags

    def fit(self, X, y):
        X_raw = X
        input_names = input_feature_names(X)
        X, y = validate_fit_data(self, X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.d_ = X.shape[1]
        self.n_features_in_ = self.d_
        self.feature_names_in_ = np.asarray(
            resolve_feature_names(X_raw, self.d_, self.feature_names), dtype=object
        )
        self.input_feature_names_in_ = (
            np.asarray(input_names, dtype=object)
            if getattr(X_raw, "columns", None) is not None
            else None
        )
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        ).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        return self.tree_.predict(np.asarray(validate_predict_data(self, X), dtype=float))

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = [format_feature_name(name) for name in self.feature_names_in_]
        return (
            f"Decision Tree Regressor (max_depth={self.max_depth}):\n"
            f"{export_text(self.tree_, feature_names=names)}"
        )
