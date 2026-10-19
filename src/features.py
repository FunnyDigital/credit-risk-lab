"""Weight-of-Evidence (WoE) binning and Information Value (IV).

This is the single most important feature-engineering technique in retail
credit risk. Every raw variable is binned, and each bin is replaced by its
WoE:

    WoE_bin = ln( %Good_in_bin / %Bad_in_bin )

where "Good" = did not default, "Bad" = defaulted.

Why bother? Three reasons a risk analyst cares about:

1. It linearises the relationship between the feature and the log-odds of
   default, which is exactly what logistic regression assumes.
2. It handles missing values and outliers natively (they just become their
   own bin).
3. It makes the model monotonic and *explainable* - a regulator can read it.

The Information Value of a feature is the sum of its per-bin contributions:

    IV = sum_bin ( %Good - %Bad ) * WoE

Rule of thumb: IV < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium,
0.3-0.5 strong, > 0.5 suspicious (possible leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


class WoEBinner:
    """Fit a binning scheme on training data, then transform any frame to WoE."""

    def __init__(self, max_bins: int = 8, min_bin_pct: float = 0.03,
                 smoothing: float = 0.5):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.smoothing = smoothing
        self.columns_: list[str] = []
        self.numeric_: dict[str, bool] = {}
        self.edges_: dict[str, np.ndarray] = {}
        self.rare_: dict[str, set] = {}
        self.woe_maps_: dict[str, dict] = {}
        self.fallback_: dict[str, float] = {}
        self.descriptions_: dict[str, dict] = {}
        self.tables_: dict[str, pd.DataFrame] = {}
        self.iv_: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        self.columns_ = list(X.columns)
        y = pd.Series(np.asarray(y).ravel())
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for col in self.columns_:
            if _is_numeric(X[col]):
                self.numeric_[col] = True
                self._fit_numeric(col, X[col], y)
            else:
                self.numeric_[col] = False
                self._fit_categorical(col, X[col], y)
        return self

    def _fit_numeric(self, col: str, s: pd.Series, y: pd.Series) -> None:
        s = pd.to_numeric(s, errors="coerce")
        non_null = s.dropna()

        if non_null.nunique() <= 1:
            edges = np.array([-np.inf, np.inf])
        else:
            qs = np.nanquantile(non_null, np.linspace(0, 1, self.max_bins + 1))
            inner = np.unique(qs[1:-1])
            edges = np.concatenate(([-np.inf], inner, [np.inf]))
            edges = self._reduce_bins(edges, non_null)

        self.edges_[col] = edges
        labels = self._apply_numeric(col, s)
        table = self._woe_table(labels, y)

        self.woe_maps_[col] = table["woe"].to_dict()
        self.fallback_[col] = 0.0
        self.tables_[col] = table
        self.iv_[col] = float(table["iv"].sum())
        self.descriptions_[col] = {
            lab: self._numeric_desc(col, lab, edges) for lab in table.index
        }

    def _reduce_bins(self, edges: np.ndarray, values: pd.Series) -> np.ndarray:
        """Merge away interior bins that are too small (rare-bin pooling)."""
        threshold = self.min_bin_pct * len(values)
        while len(edges) > 2:
            codes = pd.cut(values, bins=edges, include_lowest=True, labels=False)
            counts = codes.value_counts()
            smallest = min(
                range(len(edges) - 1), key=lambda i: counts.get(i, 0)
            )
            if counts.get(smallest, 0) >= threshold:
                break
            drop_at = 1 if smallest == 0 else smallest
            edges = np.delete(edges, drop_at)
        return edges

    def _fit_categorical(self, col: str, s: pd.Series, y: pd.Series) -> None:
        cat = s.astype("object").where(s.notna(), "MISSING").astype(str)
        freq = cat.value_counts(normalize=True)
        rare = set(freq[freq < self.min_bin_pct].index) - {"MISSING"}
        grouped = cat.where(~cat.isin(rare), "OTHER")

        self.rare_[col] = rare
        table = self._woe_table(grouped, y)
        self.woe_maps_[col] = table["woe"].to_dict()
        self.fallback_[col] = float(table["woe"].get("OTHER", 0.0))
        self.tables_[col] = table
        self.iv_[col] = float(table["iv"].sum())
        self.descriptions_[col] = {lab: str(lab) for lab in table.index}

    def _apply_numeric(self, col: str, s: pd.Series) -> pd.Series:
        codes = pd.cut(s, bins=self.edges_[col], include_lowest=True, labels=False)
        return pd.Series(
            np.where(codes.isna(), "MISSING", "bin_" + codes.astype("Int64").astype(str)),
            index=s.index,
        )

    def _woe_table(self, labels: pd.Series, y: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame({"bin": labels.values, "bad": y.values})
        agg = frame.groupby("bin", dropna=False)["bad"].agg(["count", "sum"])
        agg = agg.rename(columns={"sum": "bad"})
        agg["good"] = agg["count"] - agg["bad"]

        total_good = agg["good"].sum()
        total_bad = agg["bad"].sum()
        k = self.smoothing
        n_bins = len(agg)
        agg["bad_dist"] = (agg["bad"] + k) / (total_bad + k * n_bins)
        agg["good_dist"] = (agg["good"] + k) / (total_good + k * n_bins)
        agg["woe"] = np.log(agg["good_dist"] / agg["bad_dist"])
        agg["bad_rate"] = agg["bad"] / agg["count"]
        agg["iv"] = (agg["good_dist"] - agg["bad_dist"]) * agg["woe"]
        return agg

    @staticmethod
    def _numeric_desc(col: str, label: str, edges: np.ndarray) -> str:
        if label == "MISSING":
            return f"{col} is missing"
        i = int(label.split("_")[1])
        lo, hi = edges[i], edges[i + 1]
        if np.isneginf(lo):
            return f"{col} < {hi:,.4g}"
        if np.isposinf(hi):
            return f"{col} >= {lo:,.4g}"
        return f"{lo:,.4g} <= {col} < {hi:,.4g}"

    # -------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.columns_).reset_index(drop=True)
        out = {}
        for col in self.columns_:
            if self.numeric_[col]:
                labels = self._apply_numeric(col, pd.to_numeric(X[col], errors="coerce"))
            else:
                cat = X[col].astype("object").where(X[col].notna(), "MISSING").astype(str)
                labels = cat.where(~cat.isin(self.rare_[col]), "OTHER")
            out[col] = labels.map(self.woe_maps_[col]).fillna(self.fallback_[col])
        return pd.DataFrame(out)

    # ------------------------------------------------------------- reporting
    def bin_label(self, col: str, value) -> str:
        if self.numeric_[col]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "MISSING"
            codes = pd.cut(
                pd.Series([value]), bins=self.edges_[col], include_lowest=True,
                labels=False,
            )
            if pd.isna(codes.iloc[0]):
                return "MISSING"
            return f"bin_{int(codes.iloc[0])}"
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "MISSING"
        value = str(value)
        if value in self.rare_[col]:
            return "OTHER"
        return value if value in self.woe_maps_[col] else "MISSING"

    def describe(self, col: str, label: str) -> str:
        return self.descriptions_[col].get(label, label)

    def iv_summary(self) -> pd.DataFrame:
        df = (
            pd.DataFrame({"feature": list(self.iv_.keys()),
                          "iv": list(self.iv_.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        df["strength"] = pd.cut(
            df["iv"],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "suspicious"],
            right=False,
        ).astype(str)
        return df

    def binning_table(self) -> pd.DataFrame:
        frames = []
        for col, table in self.tables_.items():
            t = table.copy()
            t.insert(0, "feature", col)
            t.insert(1, "label", t.index)
            t["description"] = [self.describe(col, lab) for lab in t.index]
            t = t.reset_index(drop=True)
            t["bin_share"] = t["count"] / t["count"].sum()
            frames.append(t)
        cols = ["feature", "label", "description", "count", "bin_share",
                "bad", "good", "bad_rate", "woe", "iv"]
        return pd.concat(frames, ignore_index=True)[cols]
