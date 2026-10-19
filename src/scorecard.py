"""Logistic-regression scorecard with points scaling.

The classic retail scorecard is:

    score = offset - factor * logit(default)

with

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

so that a score of ``base_score`` corresponds to ``base_odds`` good:bad, and
every extra ``PDO`` points doubles the odds of being good. Because the input
Features are WoE values, each bin of each variable carries a fixed number of
points - which is why a scorecard can literally be printed as a table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config
from .features import WoEBinner


class Scorecard:
    def __init__(
        self,
        binner: WoEBinner,
        base_score: float = config.BASE_SCORE,
        base_odds: float = config.BASE_ODDS,
        pdo: float = config.PDO,
        C: float = 1.0,
    ):
        self.binner = binner
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.C = C

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "Scorecard":
        Xw = self.binner.transform(X)
        self.features_ = list(Xw.columns)
        self.model_ = LogisticRegression(C=self.C, max_iter=5000, solver="lbfgs")
        self.model_.fit(Xw, np.asarray(y).ravel())

        self.factor_ = self.pdo / np.log(2.0)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        self.base_points_ = self.offset_ - self.factor_ * float(self.model_.intercept_[0])
        self.points_table_ = self._points_table()
        return self

    def _points_table(self) -> pd.DataFrame:
        rows = []
        coefs = dict(zip(self.features_, self.model_.coef_[0]))
        for feat in self.features_:
            table = self.binner.tables_[feat]
            for label, row in table.iterrows():
                points = -self.factor_ * coefs[feat] * float(row["woe"])
                rows.append({
                    "feature": feat,
                    "label": label,
                    "description": self.binner.describe(feat, label),
                    "woe": float(row["woe"]),
                    "coefficient": coefs[feat],
                    "points": points,
                    "bin_share": float(row["count"] / table["count"].sum()),
                    "bad_rate": float(row["bad_rate"]),
                })
        return pd.DataFrame(rows)

    def card_summary(self) -> pd.DataFrame:
        """Points grouped by feature, with the score swing each variable owns."""
        g = self.points_table_.groupby("feature")["points"]
        out = pd.DataFrame({"min_points": g.min(), "max_points": g.max()})
        out["swing"] = out["max_points"] - out["min_points"]
        return out.sort_values("swing", ascending=False).reset_index()

    # -------------------------------------------------------------- predict
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        Xw = self.binner.transform(X)
        return self.model_.predict_proba(Xw)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        Xw = self.binner.transform(X)
        logit = self.model_.decision_function(Xw)
        return self.offset_ - self.factor_ * logit

    def explain(self, X: pd.DataFrame, top: int = 6) -> list[dict]:
        """Reason codes for a single applicant, worst contribution first."""
        Xw = self.binner.transform(X)
        row = Xw.iloc[0]
        raw = X.iloc[0]
        contribs = []
        for j, feat in enumerate(self.features_):
            woe = float(row[feat])
            points = -self.factor_ * float(self.model_.coef_[0][j]) * woe
            label = self.binner.bin_label(feat, raw[feat])
            contribs.append({
                "feature": feat,
                "value": raw[feat],
                "bin": label,
                "description": self.binner.describe(feat, label),
                "woe": round(woe, 4),
                "points": round(points, 2),
            })
        contribs.sort(key=lambda d: d["points"])
        return contribs[:top]
