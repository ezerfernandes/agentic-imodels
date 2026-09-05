"""hinge_gam — HingeGAMRegressor from the agentic-imodels library.

Generated from: result_libs/apr9-claude-effort=medium-main-result/
    interpretable_regressors_lib/failure/interpretable_regressor_d551a55_hinge_gam_10bp.py

Shorthand: HingeGAM_10bp
Mean global rank (lower is better): 280.18   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.558
    test (157 tests): 0.783

Metrics predate the exact-equation display fix in this package.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

from ._names import (
    align_feature_names,
    format_feature_name,
    input_feature_names,
    resolve_feature_names,
    validate_fit_data,
    validate_predict_data,
)


class HingeGAMRegressor(RegressorMixin, BaseEstimator):
    """
    HingeGAM: Lasso on a hinge basis (piecewise-linear), displayed as its
    exact prediction equation.

    Stage 1: LassoCV on original features + positive hinges at quantile knots.

    Display and prediction both use the fitted Lasso hinge basis directly.
    No residual EBM is used.
    """

    def __init__(
        self,
        n_knots=2,
        max_input_features=15,
        ebm_outer_bags=2,
        ebm_max_rounds=500,
        feature_names=None,
    ):
        self.n_knots = n_knots
        self.max_input_features = max_input_features
        self.ebm_outer_bags = ebm_outer_bags
        self.ebm_max_rounds = ebm_max_rounds
        self.feature_names = feature_names

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags

    def fit(self, X, y):
        X_raw = X
        input_names = input_feature_names(X)
        X, y = validate_fit_data(self, X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(
            resolve_feature_names(X_raw, self.n_features_in_, self.feature_names), dtype=object
        )
        self.input_feature_names_in_ = (
            np.asarray(input_names, dtype=object)
            if getattr(X_raw, "columns", None) is not None
            else None
        )
        n_samples, n_orig = X.shape

        # Feature selection
        if n_orig > self.max_input_features:
            corrs = np.array(
                [
                    abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-10 else 0
                    for j in range(n_orig)
                ]
            )
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features :])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Build hinge basis (positive hinges only for cleaner shape functions)
        quantiles = np.linspace(0.25, 0.75, self.n_knots)
        self.knot_info_ = []  # (feat_idx_in_sel, knot)
        basis_cols = [X_sel]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                basis_cols.append(np.maximum(0, xj - t).reshape(-1, 1))
                self.knot_info_.append((i, t))

        X_basis = np.hstack(basis_cols)

        # Fit Lasso
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_basis, y)

        self.ebm_ = None  # No EBM — pure Lasso for consistent display/predict

        # Compute per-feature shape functions from the Lasso model
        coefs = self.lasso_.coef_
        self.intercept_ = float(self.lasso_.intercept_)
        self.shape_functions_ = {}
        self.feature_importances_ = np.zeros(n_orig)

        for i_sel in range(n_feat):
            j_orig = self.selected_[i_sel]
            xj = X_sel[:, i_sel]
            grid = np.linspace(np.min(xj), np.max(xj), 50)

            # Evaluate Lasso contribution for feature j at grid points
            vals = coefs[i_sel] * grid
            for idx, (feat_idx, knot) in enumerate(self.knot_info_):
                if feat_idx != i_sel:
                    continue
                c = coefs[n_feat + idx]
                vals = vals + c * np.maximum(0, grid - knot)

            # Center
            mean_val = np.mean(vals)
            vals_centered = vals - mean_val
            self.intercept_ += mean_val

            # Convert to piecewise-constant (digitize into 20 bins for shape function)
            if np.std(vals_centered) < 1e-8:
                continue

            # Use the actual Lasso-derived values at quantile breakpoints
            breakpoints = np.unique(np.quantile(xj, np.linspace(0.1, 0.9, 10)))
            intervals = []
            for bp_idx in range(len(breakpoints) + 1):
                if bp_idx == 0:
                    test_x = breakpoints[0] - 0.5
                elif bp_idx == len(breakpoints):
                    test_x = breakpoints[-1] + 0.5
                else:
                    test_x = (breakpoints[bp_idx - 1] + breakpoints[bp_idx]) / 2
                # Evaluate Lasso at this point
                v = coefs[i_sel] * test_x
                for idx, (feat_idx, knot) in enumerate(self.knot_info_):
                    if feat_idx != i_sel:
                        continue
                    c = coefs[n_feat + idx]
                    v += c * max(0, test_x - knot)
                intervals.append(v - mean_val)

            # Laplacian smoothing (3 passes)
            if len(intervals) > 2:
                smooth = list(intervals)
                for _ in range(3):
                    new_s = [smooth[0]]
                    for k in range(1, len(smooth) - 1):
                        new_s.append(0.6 * smooth[k] + 0.2 * smooth[k - 1] + 0.2 * smooth[k + 1])
                    new_s.append(smooth[-1])
                    smooth = new_s
                intervals = smooth

            self.shape_functions_[j_orig] = (list(breakpoints), intervals)
            self.feature_importances_[j_orig] = max(intervals) - min(intervals)

        # Linear approximation per feature
        self.linear_approx_ = {}
        for j_orig, (thresholds, intervals) in self.shape_functions_.items():
            j_sel = np.where(self.selected_ == j_orig)[0][0]
            xj = X_sel[:, j_sel]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[min(b, len(intervals) - 1)] for b in bins])
            if np.std(xj) > 1e-10 and np.std(fx) > 1e-10:
                slope = np.cov(xj, fx)[0, 1] / np.var(xj)
                offset = np.mean(fx) - slope * np.mean(xj)
                fx_lin = slope * xj + offset
                ss_res = np.sum((fx - fx_lin) ** 2)
                ss_tot = np.sum((fx - np.mean(fx)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j_orig] = (slope, offset, r2)
            else:
                self.linear_approx_[j_orig] = (0.0, float(np.mean(fx)), 1.0)

        return self

    def _build_basis(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot in self.knot_info_:
            cols.append(np.maximum(0, X_sel[:, feat_idx] - knot).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        X = np.asarray(validate_predict_data(self, X), dtype=np.float64)
        pred = self.lasso_.predict(self._build_basis(X))
        if self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "lasso_")
        feature_names = [format_feature_name(name) for name in self.feature_names_in_]
        coefficients = np.asarray(self.lasso_.coef_, dtype=float)
        n_selected = len(self.selected_)
        n_breakpoints = len(self.knot_info_)
        n_active = int(np.count_nonzero(coefficients))
        lines = [
            f"Hinge GAM (LassoCV on linear + hinge basis, alpha={self.lasso_.alpha_:.4g}, "
            f"{n_breakpoints} breakpoints, {n_active} active terms):"
        ]
        lines.extend(
            [
                "Exact prediction equation:",
                f"  y = {float(self.lasso_.intercept_):.17g}",
            ]
        )

        active = set()
        for selected_index, original_index in enumerate(self.selected_):
            coefficient = float(coefficients[selected_index])
            if coefficient == 0.0:
                continue
            active.add(int(original_index))
            operator = "+" if coefficient > 0.0 else "-"
            lines.append(
                f"      {operator} {abs(coefficient):.17g} * {feature_names[int(original_index)]}"
            )

        for coefficient_index, (selected_index, knot) in enumerate(self.knot_info_):
            coefficient = float(coefficients[n_selected + coefficient_index])
            if coefficient == 0.0:
                continue
            original_index = int(self.selected_[selected_index])
            active.add(original_index)
            operator = "+" if coefficient > 0.0 else "-"
            lines.append(
                f"      {operator} {abs(coefficient):.17g} * "
                f"max(0, {feature_names[original_index]} - ({float(knot):.17g}))"
            )

        lines.append("")
        lines.append(
            "Active features: "
            + (", ".join(feature_names[j] for j in sorted(active)) if active else "none")
        )
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append(f"Zero-contribution features: {', '.join(inactive)}")
        return "\n".join(lines)
