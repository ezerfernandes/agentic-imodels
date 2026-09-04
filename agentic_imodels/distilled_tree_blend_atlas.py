"""distilled_tree_blend_atlas — DistilledTreeBlendAtlasRegressor from the agentic-imodels library.

Generated from: result_libs/apr19-codex-5.3-effort=xhigh/
    interpretable_regressors_lib/success/interpretable_regressor_d34b7ed_distilledtreeblendatlasapr18aa.py

Shorthand: DistilledTreeBlendAtlas_v1
Mean global rank (lower is better): 139.69   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  1.000
    test (157 tests): 0.707
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ._names import (
    align_feature_names,
    format_feature_name,
    input_feature_names,
    resolve_feature_names,
    validate_fit_data,
    validate_predict_data,
)


class DistilledTreeBlendAtlasRegressor(RegressorMixin, BaseEstimator):
    """
    Custom distilled ensemble:
      1) Ridge student on standardized features.
      2) Gradient-boosting teacher on raw features.
      3) Random-forest teacher on raw features.
      4) Validation-calibrated nonnegative blending + 1D affine calibration.

    __str__ exposes a data-driven partial-dependence card for the fitted model.
    """

    def __init__(
        self,
        student_alpha_grid=(0.02, 0.08, 0.25, 0.8, 2.5),
        validation_fraction=0.22,
        min_validation_samples=24,
        blend_l2=1e-4,
        min_component_weights=(0.12, 0.10, 0.08),  # (gbm, rf, student)
        gbm_estimators_base=110,
        gbm_estimators_scale=3.5,
        gbm_estimators_cap=190,
        gbm_learning_rate=0.05,
        gbm_max_depth=3,
        gbm_subsample=0.85,
        gbm_min_samples_leaf=5,
        rf_estimators_base=90,
        rf_estimators_scale=4.0,
        rf_estimators_cap=210,
        rf_max_depth=8,
        rf_min_samples_leaf=2,
        rf_max_features=0.7,
        calibration_slope_min=0.65,
        calibration_slope_max=1.35,
        equation_terms=8,
        inactive_rel_threshold=0.08,
        random_state=42,
        feature_names=None,
    ):
        self.student_alpha_grid = student_alpha_grid
        self.validation_fraction = validation_fraction
        self.min_validation_samples = min_validation_samples
        self.blend_l2 = blend_l2
        self.min_component_weights = min_component_weights
        self.gbm_estimators_base = gbm_estimators_base
        self.gbm_estimators_scale = gbm_estimators_scale
        self.gbm_estimators_cap = gbm_estimators_cap
        self.gbm_learning_rate = gbm_learning_rate
        self.gbm_max_depth = gbm_max_depth
        self.gbm_subsample = gbm_subsample
        self.gbm_min_samples_leaf = gbm_min_samples_leaf
        self.rf_estimators_base = rf_estimators_base
        self.rf_estimators_scale = rf_estimators_scale
        self.rf_estimators_cap = rf_estimators_cap
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features
        self.calibration_slope_min = calibration_slope_min
        self.calibration_slope_max = calibration_slope_max
        self.equation_terms = equation_terms
        self.inactive_rel_threshold = inactive_rel_threshold
        self.random_state = random_state
        self.feature_names = feature_names

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags

    def _ridge_with_intercept(self, X, y, l2):
        n = X.shape[0]
        p = X.shape[1]
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        A = np.column_stack([np.ones(n, dtype=float), X])
        reg = np.diag([0.0] + [float(l2)] * p).astype(float)
        lhs = A.T @ A + reg
        rhs = A.T @ y
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _ridge_predict(self, X, intercept, coef):
        if X.shape[1] == 0 or coef.size == 0:
            return np.full(X.shape[0], float(intercept), dtype=float)
        return float(intercept) + X @ coef

    def _fit_student(self, Xz, y, train_idx, val_idx):
        alphas = np.asarray(self.student_alpha_grid, dtype=float).reshape(-1)
        alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
        if alphas.size == 0:
            alphas = np.array([0.1], dtype=float)

        if train_idx is None or val_idx is None or len(val_idx) < 6:
            alpha = float(np.median(alphas))
            intercept, coef = self._ridge_with_intercept(Xz, y, alpha)
            return alpha, intercept, coef

        Xtr = Xz[train_idx]
        ytr = y[train_idx]
        Xva = Xz[val_idx]
        yva = y[val_idx]

        best = None
        for alpha in alphas:
            intercept, coef = self._ridge_with_intercept(Xtr, ytr, alpha)
            pred = self._ridge_predict(Xva, intercept, coef)
            rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, float(alpha), float(intercept), np.asarray(coef, dtype=float))

        return best[1], best[2], best[3]

    def _solve_nonnegative_weights(self, preds, y):
        if preds.size == 0:
            return np.array([1.0], dtype=float)

        k = preds.shape[1]
        lhs = preds.T @ preds + float(self.blend_l2) * np.eye(k, dtype=float)
        rhs = preds.T @ y
        try:
            w = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(preds, y, rcond=None)[0]

        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        floor = np.asarray(self.min_component_weights, dtype=float)
        if floor.size == k:
            w = np.maximum(w, floor)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 1e-12:
            w = np.ones(k, dtype=float) / float(k)
        else:
            w = w / total
        return w

    def _make_teachers(self, n_samples, n_features, seed):
        gbm_estimators = int(
            np.clip(
                float(self.gbm_estimators_base)
                + float(self.gbm_estimators_scale) * np.sqrt(max(n_samples, 1)),
                60,
                int(self.gbm_estimators_cap),
            )
        )
        rf_estimators = int(
            np.clip(
                float(self.rf_estimators_base)
                + float(self.rf_estimators_scale) * np.sqrt(max(n_samples, 1)),
                60,
                int(self.rf_estimators_cap),
            )
        )

        gbm_min_leaf = int(np.clip(float(self.gbm_min_samples_leaf), 1, max(1, n_samples // 8)))
        rf_min_leaf = int(np.clip(float(self.rf_min_samples_leaf), 1, max(1, n_samples // 8)))
        rf_max_features = float(np.clip(float(self.rf_max_features), 0.1, 1.0))

        gbm = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=gbm_estimators,
            learning_rate=float(self.gbm_learning_rate),
            max_depth=int(self.gbm_max_depth),
            subsample=float(self.gbm_subsample),
            min_samples_leaf=gbm_min_leaf,
            random_state=int(seed),
        )
        rf = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=int(self.rf_max_depth),
            min_samples_leaf=rf_min_leaf,
            max_features=rf_max_features,
            bootstrap=True,
            n_jobs=1,
            random_state=int(seed) + 1,
        )
        return gbm, rf

    def _term_to_str(self, term):
        if term[0] == "linear":
            return format_feature_name(self.feature_names_in_[term[1]])
        return str(term)

    def fit(self, X, y):
        X_raw = X
        input_names = input_feature_names(X)
        X, y = validate_fit_data(self, X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_in_ = np.asarray(
            resolve_feature_names(X_raw, n_features, self.feature_names), dtype=object
        )
        self.input_feature_names_in_ = (
            np.asarray(input_names, dtype=object)
            if getattr(X_raw, "columns", None) is not None
            else None
        )

        if n_features == 0:
            self.feature_quantiles_ = np.empty((3, 0), dtype=float)
            self.x_mean_ = np.zeros(0, dtype=float)
            self.x_scale_ = np.ones(0, dtype=float)
            self.student_alpha_ = 0.0
            self.student_intercept_ = float(np.mean(y))
            self.student_coef_ = np.zeros(0, dtype=float)
            self.gbm_ = None
            self.rf_ = None
            self.blend_weights_ = np.array([0.0, 0.0, 1.0], dtype=float)
            self.calibration_intercept_ = 0.0
            self.calibration_slope_ = 1.0
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(0, dtype=float)
            self.terms_ = []
            self.feature_importance_ = np.zeros(0, dtype=float)
            self.meaningful_features_ = []
            self.inactive_features_ = []
            self.training_rmse_ = float(np.sqrt(np.mean((y - self.intercept_) ** 2)))
            return self

        self.feature_quantiles_ = np.quantile(X, [0.1, 0.5, 0.9], axis=0)
        self.x_mean_ = X.mean(axis=0).astype(float)
        self.x_scale_ = X.std(axis=0).astype(float)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Xz = (X - self.x_mean_) / self.x_scale_

        train_idx = None
        val_idx = None
        if n_samples >= int(self.min_validation_samples) * 2:
            idx = np.arange(n_samples)
            tr, va = train_test_split(
                idx,
                test_size=float(self.validation_fraction),
                random_state=int(self.random_state),
            )
            if len(va) >= int(self.min_validation_samples) and len(tr) >= int(
                self.min_validation_samples
            ):
                train_idx = tr
                val_idx = va

        (
            self.student_alpha_,
            self.student_intercept_,
            self.student_coef_,
        ) = self._fit_student(Xz, y, train_idx, val_idx)

        if train_idx is not None and val_idx is not None:
            gbm_val, rf_val = self._make_teachers(
                n_samples=len(train_idx),
                n_features=n_features,
                seed=int(self.random_state),
            )
            Xtr, ytr = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]
            Xzva = Xz[val_idx]

            gbm_val.fit(Xtr, ytr)
            rf_val.fit(Xtr, ytr)
            p_val = np.column_stack(
                [
                    gbm_val.predict(Xva),
                    rf_val.predict(Xva),
                    self._ridge_predict(Xzva, self.student_intercept_, self.student_coef_),
                ]
            ).astype(float)
            self.blend_weights_ = self._solve_nonnegative_weights(p_val, yva)
        else:
            self.blend_weights_ = self._solve_nonnegative_weights(
                np.ones((1, 3), dtype=float),
                np.ones(1, dtype=float),
            )

        self.gbm_, self.rf_ = self._make_teachers(
            n_samples=n_samples,
            n_features=n_features,
            seed=int(self.random_state),
        )
        self.gbm_.fit(X, y)
        self.rf_.fit(X, y)

        p_lin = self._ridge_predict(Xz, self.student_intercept_, self.student_coef_)
        p_gbm = self.gbm_.predict(X)
        p_rf = self.rf_.predict(X)
        blend_pred = (
            float(self.blend_weights_[0]) * p_gbm
            + float(self.blend_weights_[1]) * p_rf
            + float(self.blend_weights_[2]) * p_lin
        )

        blend_centered = blend_pred - float(np.mean(blend_pred))
        y_centered = y - float(np.mean(y))
        denom = float(np.dot(blend_centered, blend_centered))
        if denom > 1e-12:
            slope = float(np.dot(blend_centered, y_centered) / denom)
        else:
            slope = 1.0
        slope = float(
            np.clip(slope, float(self.calibration_slope_min), float(self.calibration_slope_max))
        )
        intercept = float(np.mean(y) - slope * np.mean(blend_pred))
        self.calibration_intercept_ = intercept
        self.calibration_slope_ = slope

        final_pred = self.calibration_intercept_ + self.calibration_slope_ * blend_pred
        self.training_rmse_ = float(np.sqrt(np.mean((y - final_pred) ** 2)))

        raw_student_coef = self.student_coef_ / self.x_scale_
        raw_student_intercept = float(
            self.student_intercept_ - np.dot(raw_student_coef, self.x_mean_)
        )
        self.intercept_ = raw_student_intercept

        order = np.argsort(np.abs(raw_student_coef))[::-1]
        k = int(min(max(1, int(self.equation_terms)), n_features))
        selected = [int(j) for j in order[:k] if abs(raw_student_coef[j]) > 1e-10]
        if not selected:
            selected = [int(order[0])]
        self.terms_ = [("linear", j) for j in selected]
        self.coef_ = np.asarray([raw_student_coef[j] for j in selected], dtype=float)

        lin_imp = np.abs(raw_student_coef)
        lin_imp = lin_imp / max(float(lin_imp.sum()), 1e-12)
        gbm_imp = np.asarray(self.gbm_.feature_importances_, dtype=float)
        rf_imp = np.asarray(self.rf_.feature_importances_, dtype=float)
        gbm_imp = gbm_imp / max(float(gbm_imp.sum()), 1e-12)
        rf_imp = rf_imp / max(float(rf_imp.sum()), 1e-12)

        feature_importance = (
            float(self.blend_weights_[0]) * gbm_imp
            + float(self.blend_weights_[1]) * rf_imp
            + float(self.blend_weights_[2]) * lin_imp
        )
        if not np.all(np.isfinite(feature_importance)) or float(feature_importance.sum()) <= 1e-12:
            feature_importance = lin_imp

        self.feature_importance_ = np.asarray(feature_importance, dtype=float)
        max_imp = float(np.max(self.feature_importance_))
        cutoff = float(self.inactive_rel_threshold) * max(max_imp, 1e-12)
        self.meaningful_features_ = [
            format_feature_name(self.feature_names_in_[i])
            for i in range(n_features)
            if self.feature_importance_[i] >= cutoff
        ]
        self.inactive_features_ = [
            format_feature_name(self.feature_names_in_[i])
            for i in range(n_features)
            if self.feature_importance_[i] < cutoff
        ]
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "terms_",
                "feature_importance_",
                "training_rmse_",
                "n_features_in_",
                "x_mean_",
                "x_scale_",
                "student_intercept_",
                "student_coef_",
                "blend_weights_",
                "calibration_intercept_",
                "calibration_slope_",
            ],
        )
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        X = np.asarray(validate_predict_data(self, X), dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        if self.n_features_in_ == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)

        Xz = (X - self.x_mean_) / self.x_scale_
        p_lin = self._ridge_predict(Xz, self.student_intercept_, self.student_coef_)

        if self.gbm_ is None or self.rf_ is None:
            blend = p_lin
        else:
            p_gbm = self.gbm_.predict(X)
            p_rf = self.rf_.predict(X)
            blend = (
                float(self.blend_weights_[0]) * p_gbm
                + float(self.blend_weights_[1]) * p_rf
                + float(self.blend_weights_[2]) * p_lin
            )
        return self.calibration_intercept_ + self.calibration_slope_ * blend

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "terms_",
                "feature_importance_",
                "meaningful_features_",
                "inactive_features_",
                "training_rmse_",
                "n_features_in_",
            ],
        )

        ordered = (
            np.argsort(self.feature_importance_)[::-1]
            if self.n_features_in_ > 0
            else np.array([], dtype=int)
        )
        ranked = [
            format_feature_name(self.feature_names_in_[int(j)])
            for j in ordered[: min(8, len(ordered))]
        ]
        lines = [
            "Distilled Tree-Blend Atlas Regressor",
            "Predictive core: ridge student + GBM teacher + RF teacher with calibrated blending.",
            "Training-data partial dependence card for the fitted blend.",
            "blend_weights_gbm_rf_student: "
            f"{self.blend_weights_[0]:.3f}, {self.blend_weights_[1]:.3f}, "
            f"{self.blend_weights_[2]:.3f}",
            "calibration_affine: "
            f"y_final = {self.calibration_intercept_:+.6f} "
            f"+ {self.calibration_slope_:.6f} * y_blend",
            f"training_rmse: {self.training_rmse_:.6f}",
            f"active_term_count: {len(self.terms_)}",
        ]

        if ranked:
            lines.append("feature_ranking_from_fit: " + ", ".join(ranked))
        if self.meaningful_features_:
            lines.append("meaningful_features: " + ", ".join(self.meaningful_features_))
        if self.inactive_features_:
            lines.append("near_zero_features: " + ", ".join(self.inactive_features_))

        eq = [f"{self.intercept_:+.6f}"]
        for coef, term in zip(self.coef_, self.terms_, strict=True):
            eq.append(f"{float(coef):+.6f}*{self._term_to_str(term)}")
        lines.append("sparse_equation: y = " + " ".join(eq))
        lines.append("predict() returns the calibrated blend, not the sparse equation above.")

        if self.n_features_in_ > 0:
            median_row = self.feature_quantiles_[1].copy()
            for feature_idx in ordered[: min(5, self.n_features_in_)]:
                feature_idx = int(feature_idx)
                predictions = []
                for quantile_idx in (0, 1, 2):
                    row = median_row.copy()
                    row[feature_idx] = self.feature_quantiles_[quantile_idx, feature_idx]
                    predictions.append(float(self.predict(row.reshape(1, -1))[0]))
                name = format_feature_name(self.feature_names_in_[feature_idx])
                lines.append(
                    f"pd_card {name}: p10={predictions[0]:.6f} "
                    f"p50={predictions[1]:.6f} p90={predictions[2]:.6f}"
                )

        return "\n".join(lines)
