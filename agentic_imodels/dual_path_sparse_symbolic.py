"""dual_path_sparse_symbolic — DualPathSparseSymbolicRegressor from the agentic-imodels library.

Generated from: result_libs/apr17-codex-5.3-effort=high/
    interpretable_regressors_lib/failure/interpretable_regressor_4c8b421_dualpathsparsesymbolic_v2.py

Shorthand: DualPathSparseSymbolic_v2
Mean global rank (lower is better): 163.50   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.698
    test (157 tests): 0.713
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from ._names import (
    align_feature_names,
    format_feature_name,
    input_feature_names,
    resolve_feature_names,
    validate_fit_data,
    validate_predict_data,
)

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class DualPathSparseSymbolicRegressor(RegressorMixin, BaseEstimator):
    """
    Teacher/student regressor with an explicit prediction path.

    ``predict_with="teacher"`` (the default) uses the blended teacher for
    every row.  ``predict_with="student"`` uses the sparse symbolic student
    for every row.
    """

    def __init__(
        self,
        teacher_gbm_estimators=140,
        teacher_gbm_lr=0.05,
        teacher_gbm_depth=3,
        teacher_rf_estimators=120,
        teacher_rf_depth=10,
        teacher_rf_min_samples_leaf=2,
        teacher_ridge_alpha=1.0,
        teacher_val_fraction=0.2,
        symbolic_n_rows=1,
        student_screen_features=8,
        student_interaction_features=4,
        student_max_terms=6,
        student_alpha=1e-6,
        student_min_rel_gain=0.01,
        student_complexity_penalty=0.002,
        student_coef_tol=1e-8,
        student_prune_rel=0.08,
        coef_decimals=3,
        random_state=0,
        predict_with="teacher",
        feature_names=None,
    ):
        self.teacher_gbm_estimators = teacher_gbm_estimators
        self.teacher_gbm_lr = teacher_gbm_lr
        self.teacher_gbm_depth = teacher_gbm_depth
        self.teacher_rf_estimators = teacher_rf_estimators
        self.teacher_rf_depth = teacher_rf_depth
        self.teacher_rf_min_samples_leaf = teacher_rf_min_samples_leaf
        self.teacher_ridge_alpha = teacher_ridge_alpha
        self.teacher_val_fraction = teacher_val_fraction
        self.symbolic_n_rows = symbolic_n_rows
        self.student_screen_features = student_screen_features
        self.student_interaction_features = student_interaction_features
        self.student_max_terms = student_max_terms
        self.student_alpha = student_alpha
        self.student_min_rel_gain = student_min_rel_gain
        self.student_complexity_penalty = student_complexity_penalty
        self.student_coef_tol = student_coef_tol
        self.student_prune_rel = student_prune_rel
        self.coef_decimals = coef_decimals
        self.random_state = random_state
        self.predict_with = predict_with
        self.feature_names = feature_names

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _solve_ridge_with_intercept(D, y, alpha):
        n, p = D.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        scale = np.std(D, axis=0).astype(float)
        scale[scale < 1e-12] = 1.0
        Ds = D / scale

        A = np.column_stack([np.ones(n, dtype=float), Ds])
        reg = np.eye(p + 1, dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)

        intercept = float(sol[0])
        coef = np.asarray(sol[1:], dtype=float) / scale
        return intercept, coef

    def _split_indices(self, n_samples, val_fraction):
        if n_samples <= 80:
            idx = np.arange(n_samples, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_samples)
        n_val = int(round(float(val_fraction) * n_samples))
        n_val = min(max(n_val, 24), n_samples - 24)
        return order[n_val:], order[:n_val]

    def _fit_teacher(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0], self.teacher_val_fraction)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        self.teacher_gbm_ = GradientBoostingRegressor(
            n_estimators=int(self.teacher_gbm_estimators),
            learning_rate=float(self.teacher_gbm_lr),
            max_depth=int(self.teacher_gbm_depth),
            min_samples_leaf=int(self.teacher_rf_min_samples_leaf),
            random_state=int(self.random_state),
        )
        self.teacher_rf_ = RandomForestRegressor(
            n_estimators=int(self.teacher_rf_estimators),
            max_depth=int(self.teacher_rf_depth),
            min_samples_leaf=int(self.teacher_rf_min_samples_leaf),
            max_features=0.7,
            random_state=int(self.random_state),
            n_jobs=1,
        )
        self.teacher_ridge_ = Ridge(alpha=float(self.teacher_ridge_alpha))

        self.teacher_gbm_.fit(Xtr, ytr)
        self.teacher_rf_.fit(Xtr, ytr)
        self.teacher_ridge_.fit(Xtr, ytr)

        Pva = np.column_stack(
            [
                self.teacher_gbm_.predict(Xva),
                self.teacher_rf_.predict(Xva),
                self.teacher_ridge_.predict(Xva),
            ]
        ).astype(float)
        blend_inter, blend_w = self._solve_ridge_with_intercept(Pva, yva, alpha=1e-6)

        blend_w = np.maximum(np.asarray(blend_w, dtype=float), 0.0)
        if float(np.sum(blend_w)) <= 1e-12:
            blend_w = np.array([1.0, 0.0, 0.0], dtype=float)
            blend_inter = 0.0
        else:
            blend_w = blend_w / float(np.sum(blend_w))

        self.teacher_blend_intercept_ = float(blend_inter)
        self.teacher_weights_ = blend_w
        self.teacher_val_rmse_ = self._rmse(yva, blend_inter + Pva @ blend_w)

        # Refit on full data after selecting blend weights.
        self.teacher_gbm_.fit(X, y)
        self.teacher_rf_.fit(X, y)
        self.teacher_ridge_.fit(X, y)

    def _predict_teacher(self, X):
        w = self.teacher_weights_
        return self.teacher_blend_intercept_ + (
            w[0] * self.teacher_gbm_.predict(X)
            + w[1] * self.teacher_rf_.predict(X)
            + w[2] * self.teacher_ridge_.predict(X)
        )

    @staticmethod
    def _feature_corr_order(X, target):
        yc = target - float(np.mean(target))
        yn = float(np.linalg.norm(yc)) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            x = X[:, j]
            xc = x - float(np.mean(x))
            xn = float(np.linalg.norm(xc)) + 1e-12
            scores[j] = abs(float(xc @ yc) / (xn * yn))
        return [int(i) for i in np.argsort(scores)[::-1]]

    @staticmethod
    def _eval_term(X, term):
        ttype = term["type"]
        if ttype == "linear":
            return X[:, int(term["feature"])]
        if ttype == "square":
            x = X[:, int(term["feature"])]
            return x * x
        if ttype == "hinge_pos":
            x = X[:, int(term["feature"])]
            knot = float(term["knot"])
            return np.maximum(0.0, x - knot)
        if ttype == "hinge_neg":
            x = X[:, int(term["feature"])]
            knot = float(term["knot"])
            return np.maximum(0.0, knot - x)
        if ttype == "interaction":
            a = int(term["feature_a"])
            b = int(term["feature_b"])
            return X[:, a] * X[:, b]
        raise ValueError(f"Unknown term type: {ttype}")

    def _design_matrix(self, X, terms):
        if not terms:
            return np.zeros((X.shape[0], 0), dtype=float)
        return np.column_stack([self._eval_term(X, t) for t in terms]).astype(float)

    @staticmethod
    def _term_key(term):
        t = term["type"]
        if t in {"linear", "square"}:
            return (t, int(term["feature"]))
        if t in {"hinge_pos", "hinge_neg"}:
            return (t, int(term["feature"]), float(term["knot"]))
        if t == "interaction":
            a = int(term["feature_a"])
            b = int(term["feature_b"])
            return (t, min(a, b), max(a, b))
        return (t,)

    def _candidate_terms(self, X, target):
        order = self._feature_corr_order(X, target)
        n_screen = max(1, min(int(self.student_screen_features), X.shape[1]))
        screened = order[:n_screen]

        candidates = []
        for j in screened:
            x = X[:, j]
            med = float(np.median(x))

            candidates.append({"type": "linear", "feature": int(j)})
            candidates.append({"type": "square", "feature": int(j)})
            candidates.append({"type": "hinge_pos", "feature": int(j), "knot": 0.0})
            candidates.append({"type": "hinge_neg", "feature": int(j), "knot": 0.0})

            if abs(med) > 1e-6:
                candidates.append({"type": "hinge_pos", "feature": int(j), "knot": med})
                candidates.append({"type": "hinge_neg", "feature": int(j), "knot": med})

        n_int = max(1, min(int(self.student_interaction_features), len(screened)))
        int_feats = screened[:n_int]
        for i in range(len(int_feats)):
            for k in range(i + 1, len(int_feats)):
                a = int(int_feats[i])
                b = int(int_feats[k])
                candidates.append({"type": "interaction", "feature_a": a, "feature_b": b})

        dedup = []
        seen = set()
        for t in candidates:
            key = self._term_key(t)
            if key not in seen:
                seen.add(key)
                dedup.append(t)
        return dedup

    @staticmethod
    def _complexity_cost(terms):
        feats = set()
        for t in terms:
            if t["type"] in {"linear", "square", "hinge_pos", "hinge_neg"}:
                feats.add(int(t["feature"]))
            elif t["type"] == "interaction":
                feats.add(int(t["feature_a"]))
                feats.add(int(t["feature_b"]))
        return len(terms) + 0.5 * len(feats)

    def _fit_student(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0], self.teacher_val_fraction)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        candidates = self._candidate_terms(Xtr, ytr)
        if not candidates:
            return {
                "intercept": float(np.mean(y)),
                "terms": [],
                "validation_rmse": self._rmse(yva, np.repeat(float(np.mean(ytr)), len(yva))),
            }

        selected_idx = []
        remaining = list(range(len(candidates)))

        base_inter = float(np.mean(ytr))
        base_rmse = self._rmse(yva, np.repeat(base_inter, len(yva)))
        best_rmse = base_rmse

        max_terms = max(1, int(self.student_max_terms))
        min_gain = float(self.student_min_rel_gain)

        while remaining and len(selected_idx) < max_terms:
            best_i = None
            best_score = np.inf
            best_candidate_rmse = None

            for i in remaining:
                trial_idx = selected_idx + [i]
                trial_terms = [candidates[k] for k in trial_idx]

                Dtr = self._design_matrix(Xtr, trial_terms)
                Dva = self._design_matrix(Xva, trial_terms)
                inter, coef = self._solve_ridge_with_intercept(
                    Dtr, ytr, alpha=float(self.student_alpha)
                )
                rmse = self._rmse(yva, inter + Dva @ coef)

                comp = self._complexity_cost(trial_terms)
                score = rmse + float(self.student_complexity_penalty) * comp
                if score < best_score:
                    best_score = score
                    best_i = i
                    best_candidate_rmse = rmse

            if best_i is None:
                break

            rel_gain = (best_rmse - best_candidate_rmse) / max(best_rmse, 1e-12)
            if rel_gain < min_gain and len(selected_idx) > 0:
                break

            selected_idx.append(best_i)
            remaining.remove(best_i)
            best_rmse = best_candidate_rmse

        if not selected_idx:
            selected_idx = [0]

        selected_terms = [dict(candidates[i]) for i in selected_idx]
        Dfull = self._design_matrix(X, selected_terms)
        inter_full, coef_full = self._solve_ridge_with_intercept(
            Dfull, y, alpha=float(self.student_alpha)
        )

        coef_full = np.asarray(coef_full, dtype=float)
        max_abs = float(np.max(np.abs(coef_full))) if coef_full.size else 0.0
        keep = np.ones(coef_full.shape[0], dtype=bool)
        if max_abs > 0:
            keep &= np.abs(coef_full) >= float(self.student_prune_rel) * max_abs
        keep &= np.abs(coef_full) > float(self.student_coef_tol)

        if not np.any(keep):
            keep[np.argmax(np.abs(coef_full))] = True

        terms = []
        dec = int(self.coef_decimals)
        for term, c, k in zip(selected_terms, coef_full, keep, strict=True):
            if not k:
                continue
            t = dict(term)
            t["coef"] = float(np.round(float(c), dec))
            if abs(t["coef"]) > float(self.student_coef_tol):
                terms.append(t)

        if not terms and X.shape[1] > 0:
            terms = [{"type": "linear", "feature": 0, "coef": 0.0}]

        return {
            "intercept": float(np.round(float(inter_full), int(self.coef_decimals))),
            "terms": terms,
            "validation_rmse": float(best_rmse),
        }

    def _predict_student(self, X):
        pred = np.repeat(float(self.student_intercept_), X.shape[0]).astype(float)
        if self.student_terms_:
            D = self._design_matrix(X, self.student_terms_)
            coef = np.asarray([float(t["coef"]) for t in self.student_terms_], dtype=float)
            pred = pred + D @ coef
        return pred

    @staticmethod
    def _term_features(term):
        ttype = term["type"]
        if ttype in {"linear", "square", "hinge_pos", "hinge_neg"}:
            return {int(term["feature"])}
        if ttype == "interaction":
            return {int(term["feature_a"]), int(term["feature_b"])}
        return set()

    def fit(self, X, y):
        if self.predict_with not in {"teacher", "student"}:
            raise ValueError("predict_with must be either 'teacher' or 'student'.")
        if self.symbolic_n_rows != 1:
            warnings.warn(
                "symbolic_n_rows is deprecated and ignored; use predict_with instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        X_raw = X
        input_names = input_feature_names(X)
        X, y = validate_fit_data(self, X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = int(X.shape[1])
        self.feature_names_in_ = np.asarray(
            resolve_feature_names(X_raw, self.n_features_in_, self.feature_names), dtype=object
        )
        self.input_feature_names_in_ = (
            np.asarray(input_names, dtype=object)
            if getattr(X_raw, "columns", None) is not None
            else None
        )

        self._fit_teacher(X, y)
        student = self._fit_student(X, y)
        self.student_intercept_ = float(student["intercept"])
        self.student_terms_ = list(student["terms"])
        self.student_val_rmse_ = float(student["validation_rmse"])

        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
        for term in self.student_terms_:
            w = abs(float(term["coef"]))
            feats = sorted(self._term_features(term))
            if not feats:
                continue
            share = w / len(feats)
            for f in feats:
                self.feature_importance_[int(f)] += share
            if term["type"] == "linear":
                self.coef_[int(term["feature"])] += float(term["coef"])

        self.selected_features_ = sorted(
            int(i) for i in np.where(self.feature_importance_ > float(self.student_coef_tol))[0]
        )
        self.feature_order_ = [int(i) for i in np.argsort(self.feature_importance_)[::-1]]
        return self

    def _prepare_predict_input(self, X):
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        return np.asarray(validate_predict_data(self, X), dtype=float)

    def _check_predict_fitted(self):
        check_is_fitted(
            self,
            [
                "teacher_gbm_",
                "teacher_rf_",
                "teacher_ridge_",
                "teacher_blend_intercept_",
                "teacher_weights_",
                "teacher_val_rmse_",
                "student_intercept_",
                "student_terms_",
            ],
        )

    def predict_student(self, X):
        self._check_predict_fitted()
        return self._predict_student(self._prepare_predict_input(X))

    def predict_teacher(self, X):
        self._check_predict_fitted()
        return self._predict_teacher(self._prepare_predict_input(X))

    def predict(self, X):
        self._check_predict_fitted()
        if self.predict_with == "student":
            return self.predict_student(X)
        return self.predict_teacher(X)

    def _term_text(self, term, dec):
        ttype = term["type"]
        if ttype == "linear":
            return format_feature_name(self.feature_names_in_[int(term["feature"])])
        if ttype == "square":
            name = format_feature_name(self.feature_names_in_[int(term["feature"])])
            return f"({name}^2)"
        if ttype == "hinge_pos":
            feat = int(term["feature"])
            knot = float(term["knot"])
            name = format_feature_name(self.feature_names_in_[feat])
            return f"max(0, {name} - {knot:.{dec}f})"
        if ttype == "hinge_neg":
            feat = int(term["feature"])
            knot = float(term["knot"])
            name = format_feature_name(self.feature_names_in_[feat])
            return f"max(0, {knot:.{dec}f} - {name})"
        if ttype == "interaction":
            a = int(term["feature_a"])
            b = int(term["feature_b"])
            left = format_feature_name(self.feature_names_in_[a])
            right = format_feature_name(self.feature_names_in_[b])
            return f"({left}*{right})"
        return "0"

    def __str__(self):
        check_is_fitted(
            self,
            [
                "student_intercept_",
                "student_terms_",
                "selected_features_",
                "feature_importance_",
                "student_val_rmse_",
                "teacher_val_rmse_",
                "teacher_weights_",
            ],
        )
        dec = int(self.coef_decimals)
        terms_sorted = sorted(
            self.student_terms_, key=lambda t: abs(float(t["coef"])), reverse=True
        )

        eq = [f"{self.student_intercept_:.{dec}f}"]
        for t in terms_sorted:
            c = float(t["coef"])
            sign = "+" if c >= 0 else "-"
            eq.append(f"{sign} {abs(c):.{dec}f}*{self._term_text(t, dec)}")

        active = sorted(set(int(i) for i in self.selected_features_))
        inactive = [
            format_feature_name(self.feature_names_in_[i])
            for i in range(self.n_features_in_)
            if i not in active
        ]
        ranked_active = [
            format_feature_name(self.feature_names_in_[i])
            for i in self.feature_order_
            if i in active
        ]

        lines = [
            "Dual-Path Sparse Symbolic Regressor",
            "Display equation (sparse student). predict() uses: "
            + (
                "teacher ensemble (GBM/RF/Ridge)"
                if self.predict_with == "teacher"
                else "this equation"
            ),
            "Student-vs-teacher validation RMSE: "
            f"{self.student_val_rmse_:.6f} vs {self.teacher_val_rmse_:.6f}.",
            "  y = " + " ".join(eq),
            "Interpretation notes:",
            "  - Any omitted feature value is treated as 0.",
            "  - Any feature not listed in the equation has zero contribution.",
            "  - max(0, a) means: use a if a > 0, else use 0.",
            "  - For change questions, compute prediction(new input) minus prediction(old input).",
            "Active features (most to least influential): "
            f"{', '.join(ranked_active) if ranked_active else '(none)'}",
            f"Zero-effect features: {', '.join(inactive) if inactive else '(none)'}",
        ]
        return "\n".join(lines)
