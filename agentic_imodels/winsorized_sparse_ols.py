"""winsorized_sparse_ols — WinsorizedSparseOLSRegressor from the agentic-imodels library.

Generated from: result_libs/apr19-claude-4.7-effort=medium-rerun2/
    interpretable_regressors_lib/failure/interpretable_regressor_f53ad88_WinsorizedSparseOLS.py

Shorthand: WinsorizedSparseOLS
Mean global rank (lower is better): 326.95   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.651
    test (157 tests): 0.726
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ._names import (
    align_feature_names,
    format_feature_name,
    input_feature_names,
    resolve_feature_names,
    validate_fit_data,
    validate_predict_data,
)


class WinsorizedSparseOLSRegressor(RegressorMixin, BaseEstimator):
    """Clip features to [p1,p99] before LassoCV select top-8 + OLS refit."""

    def __init__(self, max_features=8, cv=3, low=1, high=99, feature_names=None):
        self.max_features = max_features
        self.cv = cv
        self.low = low
        self.high = high
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
        n, d = X.shape
        self.feature_names_in_ = np.asarray(
            resolve_feature_names(X_raw, d, self.feature_names), dtype=object
        )
        self.input_feature_names_in_ = (
            np.asarray(input_names, dtype=object)
            if getattr(X_raw, "columns", None) is not None
            else None
        )
        self.lo_ = np.percentile(X, self.low, axis=0)
        self.hi_ = np.percentile(X, self.high, axis=0)
        Xc = np.clip(X, self.lo_, self.hi_)
        sc = StandardScaler().fit(Xc)
        Xs = sc.transform(Xc)
        lasso = LassoCV(cv=self.cv, n_alphas=20, max_iter=5000).fit(Xs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_features]
        if not kept:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(Xc[:, self.support_], y)
        self.ols_coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        self.d_ = d
        self.n_features_in_ = d
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_coef_")
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        X = np.asarray(validate_predict_data(self, X), dtype=float)
        Xc = np.clip(X, self.lo_, self.hi_)
        return Xc[:, self.support_] @ self.ols_coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "ols_coef_")
        terms = " + ".join(
            f"{c:+.4g}*{format_feature_name(self.feature_names_in_[j])}"
            for c, j in zip(self.ols_coef_, self.support_, strict=True)
        )
        excluded = [
            format_feature_name(self.feature_names_in_[j])
            for j in range(self.d_)
            if j not in self.support_
        ]
        lines = [
            "SparseLinearOLS: y = intercept + sum_j c_j * x_j  (linear, sparse)",
            f"  y = {self.intercept_:+.4f} + {terms}",
            "",
            "Coefficients:",
            f"  intercept: {self.intercept_:.4f}",
        ]
        for c, j in zip(self.ols_coef_, self.support_, strict=True):
            lines.append(f"  {format_feature_name(self.feature_names_in_[j])}: {c:.4f}")
        if excluded:
            lines.append(f"Features excluded (zero effect): {', '.join(excluded)}")
        return "\n".join(lines)
