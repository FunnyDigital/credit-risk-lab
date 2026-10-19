"""Gradient-boosted challenger model.

In a bank you almost always run two models side by side:

* the **scorecard** - simple, monotonic, regulated, the "incumbent";
* a **challenger** - usually gradient boosting - which squeezes out extra
  ranking power (higher Gini/KS) but needs explanation tooling (SHAP,
  permutation importance) to be defensible.

We use scikit-learn's HistGradientBoostingClassifier so there is no heavyweight
dependency. SHAP is used automatically *if* it happens to be installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder


class Challenger:
    def __init__(self, random_state: int = 42, max_iter: int = 800,
                 learning_rate: float = 0.05, max_leaf_nodes: int = 63,
                 min_samples_leaf: int = 40, l2_regularization: float = 0.1):
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "Challenger":
        self.columns_ = list(X.columns)
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.cat_cols_:
            self.encoder_.fit(X[self.cat_cols_].astype(str))
            self.feature_names_ = self.num_cols_ + list(
                self.encoder_.get_feature_names_out(self.cat_cols_)
            )
        else:
            self.feature_names_ = list(self.num_cols_)

        Xt = self._transform(X)
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=self.random_state,
        )
        self.model_.fit(Xt, np.asarray(y).ravel())
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.columns_)
        num = X[self.num_cols_].astype(float).to_numpy()
        if self.cat_cols_:
            cat = self.encoder_.transform(X[self.cat_cols_].astype(str))
            return np.hstack([num, cat])
        return num

    # -------------------------------------------------------------- predict
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._transform(X))[:, 1]

    # ----------------------------------------------------------- importance
    def feature_importance(self, X: pd.DataFrame, y, n_repeats: int = 5) -> pd.DataFrame:
        result = permutation_importance(
            self.model_, self._transform(X), np.asarray(y).ravel(),
            scoring="roc_auc", n_repeats=n_repeats,
            random_state=self.random_state,
        )
        return (
            pd.DataFrame({
                "feature": self.feature_names_,
                "importance": result.importances_mean,
                "std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def shap_values(self, X: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame | None:
        """Mean |SHAP| per feature, or None if shap is unavailable."""
        try:
            import shap
        except Exception:
            return None
        try:
            sample = X.iloc[:max_rows]
            explainer = shap.TreeExplainer(self.model_)
            values = explainer.shap_values(self._transform(sample))
            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]
            return (
                pd.DataFrame({
                    "feature": self.feature_names_,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                })
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
        except Exception:
            return None
