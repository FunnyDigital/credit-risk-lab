"""Matplotlib figures for the validation report (headless / Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config  # noqa: E402

GREEN = "#10b981"
RED = "#ef4444"
INK = "#0f172a"
MUTED = "#64748b"


def _save(fig, name: str) -> str:
    path = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def plot_score_distribution(scores, y, name: str = "score_distribution.png") -> str:
    scores = np.asarray(scores)
    y = np.asarray(y).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y == 0], bins=40, color=GREEN, alpha=0.6, label="Good (no default)")
    ax.hist(scores[y == 1], bins=40, color=RED, alpha=0.6, label="Bad (default)")
    _style(ax, "Scorecard score distribution", "Score", "Applicants")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_roc(y, models: dict[str, np.ndarray], name: str = "roc_curves.png") -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, pd_scores in models.items():
        fpr, tpr, _ = roc_curve(np.asarray(y).ravel(), np.asarray(pd_scores).ravel())
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    _style(ax, "ROC curves", "False positive rate", "True positive rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, name)


def plot_ks(curve: dict, name: str = "ks_plot.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, 1, len(curve["cum_good"]))
    ax.plot(x, curve["cum_bad"], color=RED, linewidth=2, label="Bad (default)")
    ax.plot(x, curve["cum_good"], color=GREEN, linewidth=2, label="Good")
    ax.annotate(f"KS = {curve['ks']:.3f}",
                xy=(0.6, 0.55), color=INK, fontsize=11, fontweight="bold")
    _style(ax, "Kolmogorov-Smirnov separation", "Cumulative population (by score)",
           "Cumulative share")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_gains(y, models: dict[str, np.ndarray], name: str = "gains_chart.png") -> str:
    from .validation import gains_table

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, pd_scores in models.items():
        g = gains_table(y, pd_scores)
        ax.plot([0, *g["cum_pop_share"]], [0, *g["cum_bad_share"]],
                marker="o", markersize=3, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Random")
    _style(ax, "Cumulative gains (by riskiest first)",
           "Cumulative population share", "Cumulative defaults captured")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_calibration(y, models: dict[str, np.ndarray], name: str = "calibration.png") -> str:
    from .validation import calibration_table

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Perfect")
    for label, pd_scores in models.items():
        t = calibration_table(y, pd_scores)
        ax.plot(t["predicted_pd"], t["observed_pd"], marker="o", linewidth=2, label=label)
    _style(ax, "Calibration: predicted vs observed PD", "Predicted PD", "Observed PD")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_psi(table: pd.DataFrame, psi_value: float, name: str = "psi_score.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    width = 0.4
    ax.bar(x - width / 2, table["expected_share"], width, color=GREEN, label="Train (reference)")
    ax.bar(x + width / 2, table["actual_share"], width, color=RED, label="OOT (current)")
    _style(ax, f"Score PSI = {psi_value:.3f}", "Score decile", "Population share")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_importance(df: pd.DataFrame, value_col: str, title: str,
                    name: str, top: int = 15) -> str:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d) + 1.2)))
    ax.barh(d["feature"], d[value_col], color=GREEN)
    _style(ax, title, value_col, "")
    return _save(fig, name)


def plot_cutoff(tradeoff: pd.DataFrame, breakeven: float,
                name: str = "cutoff_tradeoff.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tradeoff["cutoff_pd"], tradeoff["approval_rate"], color=GREEN,
            linewidth=2, marker="o", markersize=3, label="Approval rate")
    ax.plot(tradeoff["cutoff_pd"], tradeoff["approved_bad_rate"], color=RED,
            linewidth=2, marker="o", markersize=3, label="Bad rate of approvals")
    ax.axvline(breakeven, color=MUTED, linestyle="--", linewidth=1,
               label=f"Break-even PD = {breakeven:.2f}")
    _style(ax, "Cut-off trade-off", "PD cut-off (decline above)", "Rate")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_profit(tradeoff: pd.DataFrame, name: str = "profit_curve.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tradeoff["cutoff_pd"], tradeoff["expected_profit"], color=GREEN,
            linewidth=2, marker="o", markersize=3, label="Expected profit")
    ax.plot(tradeoff["cutoff_pd"], tradeoff["realized_profit"], color=INK,
            linewidth=2, linestyle="--", marker="o", markersize=3, label="Realized profit")
    best = tradeoff.loc[tradeoff["realized_profit"].idxmax(), "cutoff_pd"]
    ax.axvline(best, color=RED, linestyle=":", linewidth=1,
               label=f"Profit-max cut-off = {best:.2f}")
    _style(ax, "Profit vs cut-off", "PD cut-off", "Portfolio profit (units of exposure)")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)
