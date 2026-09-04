"""teacher_student_rule_spline — TeacherStudentRuleSplineRegressor from the agentic-imodels library.

Generated from: result_libs/apr17-codex-5.3-effort=high/
    interpretable_regressors_lib/failure/interpretable_regressor_c2b5db4_TeacherStudentRuleSpline_v1.py

Shorthand: TeacherStudentRuleSpline_v1
Mean global rank (lower is better): 204.03   (pooled 65 dev datasets)
Interpretability (fraction passed, higher is better):
    dev  (43 tests):  0.605
    test (157 tests): 0.802
"""

import warnings
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
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


class TeacherStudentRuleSplineRegressor(RegressorMixin, BaseEstimator):
    """
    Teacher/student regressor with an explicit prediction path.

    ``predict_with="teacher"`` (the default) uses the gradient-boosted
    teacher for every row.  ``predict_with="student"`` uses the sparse
    symbolic student for every row.
    """

    def __init__(
        self,
        teacher_n_estimators=160,
        teacher_learning_rate=0.05,
        teacher_max_depth=3,
        teacher_subsample=0.9,
        val_fraction=0.2,
        max_student_features=10,
        max_student_terms=8,
        alpha_student=1e-3,
        corr_screen_rel=0.01,
        hinge_quantiles=(0.2, 0.5, 0.8),
        interaction_top_features=4,
        candidate_eval_topk=6,
        min_rel_gain=0.004,
        coef_decimals=3,
        coef_tol=1e-10,
        symbolic_n_rows=1,
        random_state=0,
        predict_with="teacher",
        feature_names=None,
    ):
        self.teacher_n_estimators = teacher_n_estimators
        self.teacher_learning_rate = teacher_learning_rate
        self.teacher_max_depth = teacher_max_depth
        self.teacher_subsample = teacher_subsample
        self.val_fraction = val_fraction
        self.max_student_features = max_student_features
        self.max_student_terms = max_student_terms
        self.alpha_student = alpha_student
        self.corr_screen_rel = corr_screen_rel
        self.hinge_quantiles = hinge_quantiles
        self.interaction_top_features = interaction_top_features
        self.candidate_eval_topk = candidate_eval_topk
        self.min_rel_gain = min_rel_gain
        self.coef_decimals = coef_decimals
        self.coef_tol = coef_tol
        self.symbolic_n_rows = symbolic_n_rows
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

    def _split_indices(self, n_samples):
        if n_samples <= 80:
            idx = np.arange(n_samples, dtype=int)
            return idx, idx

        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n_samples)
        n_val = int(round(float(self.val_fraction) * n_samples))
        n_val = min(max(n_val, 24), n_samples - 24)
        return order[n_val:], order[:n_val]

    def _fit_teacher(self, X, y):
        tr_idx, va_idx = self._split_indices(X.shape[0])
        self.teacher_model_ = GradientBoostingRegressor(
            n_estimators=int(self.teacher_n_estimators),
            learning_rate=float(self.teacher_learning_rate),
            max_depth=int(self.teacher_max_depth),
            subsample=float(self.teacher_subsample),
            random_state=int(self.random_state),
        )
        self.teacher_model_.fit(X[tr_idx], y[tr_idx])
        self.teacher_val_rmse_ = self._rmse(y[va_idx], self.teacher_model_.predict(X[va_idx]))
        self.teacher_model_.fit(X, y)
        self.teacher_name_ = (
            "GradientBoostingRegressor"
            f"(n_estimators={int(self.teacher_n_estimators)},"
            f" lr={float(self.teacher_learning_rate):.3f},"
            f" max_depth={int(self.teacher_max_depth)})"
        )

    def _screen_features(self, X, y):
        n_features = X.shape[1]
        if n_features == 0:
            return []

        yc = y - float(np.mean(y))
        xc = X - X.mean(axis=0, keepdims=True)
        corr = np.abs(xc.T @ yc)
        order = [int(i) for i in np.argsort(corr)[::-1]]
        if not order:
            return [0]

        max_corr = float(corr[order[0]])
        if max_corr <= 1e-12:
            return list(range(min(int(self.max_student_features), n_features)))

        selected = []
        for j in order:
            if corr[j] >= float(self.corr_screen_rel) * max_corr:
                selected.append(int(j))
            if len(selected) >= int(self.max_student_features):
                break

        if not selected:
            selected = [int(order[0])]
        return selected

    @staticmethod
    def _term_features(term):
        t = term["type"]
        if t in {"lin", "sq", "abs", "hinge", "step"}:
            return {int(term["feature"])}
        if t == "int":
            return {int(term["a"]), int(term["b"])}
        if t == "gate":
            return {int(term["gate"]), int(term["target"])}
        return set()

    @staticmethod
    def _eval_term(X, term):
        t = term["type"]
        if t == "lin":
            return X[:, int(term["feature"])]
        if t == "sq":
            col = X[:, int(term["feature"])]
            return col * col
        if t == "abs":
            return np.abs(X[:, int(term["feature"])])
        if t == "hinge":
            feat = int(term["feature"])
            knot = float(term["knot"])
            if int(term["direction"]) > 0:
                return np.maximum(0.0, X[:, feat] - knot)
            return np.maximum(0.0, knot - X[:, feat])
        if t == "step":
            feat = int(term["feature"])
            knot = float(term["knot"])
            return (X[:, feat] > knot).astype(float)
        if t == "int":
            return X[:, int(term["a"])] * X[:, int(term["b"])]
        if t == "gate":
            gate = int(term["gate"])
            target = int(term["target"])
            knot = float(term["knot"])
            return (X[:, gate] > knot).astype(float) * X[:, target]
        raise ValueError(f"Unknown term type: {t}")

    def _term_text(self, term, dec):
        t = term["type"]
        if t == "lin":
            return format_feature_name(self.feature_names_in_[int(term["feature"])])
        if t == "sq":
            name = format_feature_name(self.feature_names_in_[int(term["feature"])])
            return f"({name}^2)"
        if t == "abs":
            name = format_feature_name(self.feature_names_in_[int(term["feature"])])
            return f"abs({name})"
        if t == "hinge":
            feat = int(term["feature"])
            knot = float(term["knot"])
            name = format_feature_name(self.feature_names_in_[feat])
            if int(term["direction"]) > 0:
                return f"max(0, {name} - {knot:.{dec}f})"
            return f"max(0, {knot:.{dec}f} - {name})"
        if t == "step":
            feat = int(term["feature"])
            knot = float(term["knot"])
            name = format_feature_name(self.feature_names_in_[feat])
            return f"1[{name} > {knot:.{dec}f}]"
        if t == "int":
            left = format_feature_name(self.feature_names_in_[int(term["a"])])
            right = format_feature_name(self.feature_names_in_[int(term["b"])])
            return f"({left} * {right})"
        if t == "gate":
            gate = int(term["gate"])
            target = int(term["target"])
            knot = float(term["knot"])
            gate_name = format_feature_name(self.feature_names_in_[gate])
            target_name = format_feature_name(self.feature_names_in_[target])
            return f"(1[{gate_name} > {knot:.{dec}f}] * {target_name})"
        return "term"

    @staticmethod
    def _dedupe_terms(terms):
        out = []
        seen = set()
        for t in terms:
            if t["type"] in {"lin", "sq", "abs"}:
                key = (t["type"], int(t["feature"]))
            elif t["type"] in {"hinge", "step"}:
                key = (
                    t["type"],
                    int(t["feature"]),
                    round(float(t["knot"]), 6),
                    int(t.get("direction", 0)),
                )
            elif t["type"] == "int":
                a, b = sorted((int(t["a"]), int(t["b"])))
                key = ("int", a, b)
            elif t["type"] == "gate":
                key = ("gate", int(t["gate"]), int(t["target"]), round(float(t["knot"]), 6))
            else:
                key = tuple(sorted(t.items()))
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _build_candidates(self, X, screened):
        if not screened:
            return []

        terms = []
        for feat in screened:
            feat = int(feat)
            terms.append({"type": "lin", "feature": feat})
            terms.append({"type": "sq", "feature": feat})
            terms.append({"type": "abs", "feature": feat})
            xcol = X[:, feat]
            knots = [0.0]
            qvals = np.asarray(self.hinge_quantiles, dtype=float)
            if qvals.size > 0:
                knots.extend(np.quantile(xcol, qvals).tolist())
            knot_values = [
                float(k) for k in np.unique(np.asarray(knots, dtype=float)) if np.isfinite(k)
            ]
            for knot in knot_values:
                terms.append({"type": "step", "feature": feat, "knot": float(knot)})
                terms.append(
                    {"type": "hinge", "feature": feat, "knot": float(knot), "direction": 1}
                )
                terms.append(
                    {"type": "hinge", "feature": feat, "knot": float(knot), "direction": -1}
                )

        inter_feats = screened[: max(2, min(len(screened), int(self.interaction_top_features)))]
        for a, b in combinations(inter_feats, 2):
            terms.append({"type": "int", "a": int(a), "b": int(b)})

        gate_feats = inter_feats[: min(2, len(inter_feats))]
        target_feats = screened[: min(4, len(screened))]
        for gate in gate_feats:
            xg = X[:, int(gate)]
            gate_knots = [0.0, float(np.quantile(xg, 0.5))]
            for knot in gate_knots:
                if not np.isfinite(knot):
                    continue
                for target in target_feats:
                    if int(target) == int(gate):
                        continue
                    terms.append(
                        {
                            "type": "gate",
                            "gate": int(gate),
                            "target": int(target),
                            "knot": float(knot),
                        }
                    )
        return self._dedupe_terms(terms)

    def _design_matrix(self, X, terms):
        if not terms:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [self._eval_term(X, t) for t in terms]
        return np.column_stack(cols).astype(float)

    def _fit_student(self, X, y):
        n_samples, n_features = X.shape
        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        screened = self._screen_features(Xtr, ytr)
        candidates = self._build_candidates(Xtr, screened)
        if not candidates:
            return {
                "intercept": float(np.mean(y)),
                "terms": [],
                "validation_rmse": self._rmse(yva, np.repeat(float(np.mean(ytr)), len(yva))),
            }

        Dtr = self._design_matrix(Xtr, candidates)
        Dva = self._design_matrix(Xva, candidates)

        baseline_intercept = float(np.mean(ytr))
        pred_tr = np.repeat(baseline_intercept, len(ytr))
        pred_va = np.repeat(baseline_intercept, len(yva))
        best_rmse = self._rmse(yva, pred_va)

        selected = []
        for _ in range(int(self.max_student_terms)):
            residual = ytr - pred_tr
            corr_scores = []
            for j in range(Dtr.shape[1]):
                if j in selected:
                    continue
                col = Dtr[:, j]
                denom = float(np.linalg.norm(col)) + 1e-12
                corr_scores.append((abs(float(col @ residual)) / denom, j))
            if not corr_scores:
                break

            corr_scores.sort(reverse=True)
            top = [j for _, j in corr_scores[: max(1, int(self.candidate_eval_topk))]]

            cand_best = None
            for j in top:
                idx = selected + [j]
                inter, coef = self._solve_ridge_with_intercept(
                    Dtr[:, idx], ytr, alpha=float(self.alpha_student)
                )
                rmse_va = self._rmse(yva, inter + Dva[:, idx] @ coef)
                if cand_best is None or rmse_va < cand_best["rmse"]:
                    cand_best = {"j": int(j), "rmse": float(rmse_va)}
            if cand_best is None:
                break

            rel_gain = (best_rmse - float(cand_best["rmse"])) / max(best_rmse, 1e-12)
            if rel_gain < float(self.min_rel_gain):
                break

            selected.append(int(cand_best["j"]))
            inter, coef = self._solve_ridge_with_intercept(
                Dtr[:, selected], ytr, alpha=float(self.alpha_student)
            )
            pred_tr = inter + Dtr[:, selected] @ coef
            pred_va = inter + Dva[:, selected] @ coef
            best_rmse = self._rmse(yva, pred_va)

        if not selected:
            return {
                "intercept": float(np.mean(y)),
                "terms": [],
                "validation_rmse": float(best_rmse),
            }

        Dfull = self._design_matrix(X, [candidates[j] for j in selected])
        inter_full, coef_full = self._solve_ridge_with_intercept(
            Dfull, y, alpha=float(self.alpha_student)
        )

        dec = int(self.coef_decimals)
        intercept = float(np.round(inter_full, dec))
        terms = []
        for j, c in zip(selected, coef_full, strict=True):
            coef = float(np.round(float(c), dec))
            if abs(coef) <= float(self.coef_tol):
                continue
            term = dict(candidates[int(j)])
            term["coef"] = coef
            terms.append(term)

        terms.sort(key=lambda t: abs(float(t["coef"])), reverse=True)
        return {
            "intercept": intercept,
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
        self.student_validation_rmse_ = self.student_val_rmse_
        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)

        for term in self.student_terms_:
            c = abs(float(term["coef"]))
            for feat in self._term_features(term):
                self.feature_importance_[int(feat)] += c
            if term["type"] == "lin":
                self.coef_[int(term["feature"])] += float(term["coef"])

        self.selected_features_ = sorted(
            int(i) for i in np.where(self.feature_importance_ > float(self.coef_tol))[0]
        )
        return self

    def _prepare_predict_input(self, X):
        X = align_feature_names(X, self.feature_names_in_, self.input_feature_names_in_)
        return np.asarray(validate_predict_data(self, X), dtype=float)

    def _check_predict_fitted(self):
        check_is_fitted(
            self,
            [
                "teacher_model_",
                "teacher_val_rmse_",
                "student_intercept_",
                "student_terms_",
                "feature_importance_",
            ],
        )

    def predict_student(self, X):
        self._check_predict_fitted()
        return self._predict_student(self._prepare_predict_input(X))

    def predict_teacher(self, X):
        self._check_predict_fitted()
        return self.teacher_model_.predict(self._prepare_predict_input(X))

    def predict(self, X):
        self._check_predict_fitted()
        if self.predict_with == "student":
            return self.predict_student(X)
        return self.predict_teacher(X)

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
                "teacher_name_",
            ],
        )
        dec = int(self.coef_decimals)
        sorted_terms = sorted(
            self.student_terms_, key=lambda t: abs(float(t["coef"])), reverse=True
        )
        eq_terms = [f"{self.student_intercept_:+.{dec}f}"]
        for term in sorted_terms:
            eq_terms.append(f"({float(term['coef']):+.{dec}f})*{self._term_text(term, dec)}")

        lines = [
            "Teacher-Student Sparse Rule-Spline Regressor",
            "Display equation (sparse student). predict() uses: "
            + (self.teacher_name_ if self.predict_with == "teacher" else "this equation"),
            "Student-vs-teacher validation RMSE: "
            f"{self.student_val_rmse_:.6f} vs {self.teacher_val_rmse_:.6f}.",
            "",
            "  y = " + " + ".join(eq_terms),
            "",
            f"Active symbolic terms ({len(sorted_terms)} total, sorted by |coefficient|):",
        ]
        if sorted_terms:
            for k, term in enumerate(sorted_terms, 1):
                lines.append(
                    f"  t{k}: {float(term['coef']):+.{dec}f} * {self._term_text(term, dec)}"
                )
        else:
            lines.append("  (none)")

        feat_rank = sorted(
            [(i, v) for i, v in enumerate(self.feature_importance_) if v > float(self.coef_tol)],
            key=lambda t: t[1],
            reverse=True,
        )
        lines.append("")
        lines.append(
            "Feature influence ranking (sum of |term coefficients| touching each feature):"
        )
        if feat_rank:
            for i, v in feat_rank:
                name = format_feature_name(self.feature_names_in_[i])
                lines.append(f"  {name}: {float(v):.{dec}f}")
        else:
            lines.append("  (none)")

        active = sorted(set(int(i) for i in self.selected_features_))
        inactive = [
            format_feature_name(self.feature_names_in_[i])
            for i in range(self.n_features_in_)
            if i not in active
        ]
        lines.append("")
        lines.append(
            "Active features: "
            + (
                ", ".join(format_feature_name(self.feature_names_in_[i]) for i in active)
                if active
                else "(none)"
            )
        )
        lines.append("Zero-effect features: " + (", ".join(inactive) if inactive else "(none)"))
        lines.append(f"Large-batch predictor: {self.teacher_name_}")
        return "\n".join(lines)
