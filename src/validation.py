"""Model validation: the metrics a risk analyst is actually judged on.

Accuracy is close to meaningless here (a 12% default rate means "always
predict no default" scores 88% accuracy while being useless). Instead we use:

* **AUC / Gini** - ranking power.
* **KS** - the maximum separation between the good and bad score distributions.
* **PSI** - population stability between reference and current populations.
* **Calibration** - do predicted probabilities match observed default rates?
* **Lift / gains** - how much better than random the model is in the tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ks_curve(y_true, y_score) -> dict:
    df = pd.DataFrame({"y": np.asarray(y_true).ravel(), "s": np.asarray(y_score).ravel()})
    df = df.sort_values("s").reset_index(drop=True)
    n_bad = int((df["y"] == 1).sum())
    n_good = int((df["y"] == 0).sum())
    cum_bad = (df["y"] == 1).cumsum() / max(n_bad, 1)
    cum_good = (df["y"] == 0).cumsum() / max(n_good, 1)
    diff = (cum_bad - cum_good).abs()
    idx = int(diff.idxmax())
    return {
        "ks": float(diff.max()),
        "threshold": float(df["s"].iloc[idx]),
        "score": df["s"].to_numpy(),
        "cum_good": cum_good.to_numpy(),
        "cum_bad": cum_bad.to_numpy(),
    }


def ks_statistic(y_true, y_score) -> float:
    return ks_curve(y_true, y_score)["ks"]


def gini(y_true, y_score) -> float:
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def summary(y_true, y_score, name: str = "model") -> dict:
    return {
        "model": name,
        "auc": round(auc(y_true, y_score), 4),
        "gini": round(gini(y_true, y_score), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "n": int(len(y_true)),
        "bad_rate": round(float(np.mean(y_true)), 4),
    }


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> dict:
    """Population Stability Index between a reference and a comparison score."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_share = np.clip(e_counts / max(len(expected), 1), eps, None)
    a_share = np.clip(a_counts / max(len(actual), 1), eps, None)
    contrib = (a_share - e_share) * np.log(a_share / e_share)
    table = pd.DataFrame({
        "bin": range(len(e_share)),
        "expected_share": e_share,
        "actual_share": a_share,
        "contribution": contrib,
    })
    total = float(contrib.sum())
    if total < 0.10:
        band = "stable"
    elif total < 0.25:
        band = "monitor"
    else:
        band = "unstable - recalibrate"
    return {"psi": round(total, 4), "band": band, "table": table}


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true).ravel(), "p": np.asarray(y_prob).ravel()})
    df["bucket"] = pd.qcut(df["p"], q=min(bins, df["p"].nunique()),
                           duplicates="drop")
    g = df.groupby("bucket", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "predicted_pd": g["p"].mean(),
        "observed_pd": g["y"].mean(),
    }).reset_index(drop=True)
    out["gap"] = out["predicted_pd"] - out["observed_pd"]
    return out


def decile_table(y_true, y_prob) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true).ravel(), "p": np.asarray(y_prob).ravel()})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False) + 1
    total_bad = df["y"].sum()
    g = df.groupby("decile")
    out = pd.DataFrame({
        "n": g.size(),
        "bad": g["y"].sum(),
        "bad_rate": g["y"].mean(),
        "avg_pd": g["p"].mean(),
    }).reset_index()
    out["cum_bad"] = out["bad"].cumsum()
    out["cum_pop_share"] = out["n"].cumsum() / out["n"].sum()
    out["cum_bad_share"] = out["cum_bad"] / total_bad
    out["lift"] = out["bad_rate"] / df["y"].mean()
    out["cum_lift"] = out["cum_bad_share"] / out["cum_pop_share"]
    return out


def gains_table(y_true, y_prob) -> pd.DataFrame:
    t = decile_table(y_true, y_prob)
    return t[["decile", "cum_pop_share", "cum_bad_share", "cum_lift"]]


def comparison_table(models: dict[str, np.ndarray], y_true) -> pd.DataFrame:
    rows = [summary(y_true, score, name) for name, score in models.items()]
    return pd.DataFrame(rows)
