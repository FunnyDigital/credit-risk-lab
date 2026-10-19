"""From probability to decision.

A model produces a probability of default (PD). A *business* needs an action.
This module converts between PD, score and money, and picks the cut-off that
maximises expected profit rather than using an arbitrary PD threshold.

Expected profit on a loan, per unit of exposure:

    E[profit] = (1 - PD) * margin_rate - PD * LGD

Approve while that is positive. That is the whole idea behind risk-based
pricing and cut-off setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------- conversions
def _factor_offset(base_score, base_odds, pdo):
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def pd_to_score(pd_value, base_score=config.BASE_SCORE,
                base_odds=config.BASE_ODDS, pdo=config.PDO):
    """Map a probability of default to a points score (higher = safer)."""
    pd_value = np.clip(np.asarray(pd_value, dtype=float), 1e-9, 1 - 1e-9)
    factor, offset = _factor_offset(base_score, base_odds, pdo)
    odds_good = (1.0 - pd_value) / pd_value
    return offset + factor * np.log(odds_good)


def score_to_pd(score, base_score=config.BASE_SCORE,
                base_odds=config.BASE_ODDS, pdo=config.PDO):
    """Inverse of :func:`pd_to_score`."""
    factor, offset = _factor_offset(base_score, base_odds, pdo)
    odds_good = np.exp((np.asarray(score, dtype=float) - offset) / factor)
    return 1.0 / (1.0 + odds_good)


def expected_loss(pd_value, lgd: float = config.LGD, ead: float | np.ndarray = 1.0):
    return np.asarray(pd_value, dtype=float) * lgd * ead


# ------------------------------------------------------------------ cut-offs
def cutoff_table(y_true, y_prob, *, lgd: float = config.LGD,
                 margin_rate: float = config.MARGIN_RATE,
                 ead: float = 1.0, n_steps: int = 19) -> pd.DataFrame:
    """Sweep the approval cut-off and report volume, risk and profit."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    cutoffs = np.round(np.linspace(0.02, 0.80, n_steps), 3)

    rows = []
    for c in cutoffs:
        approved = y_prob < c
        n_app = int(approved.sum())
        if n_app == 0:
            rows.append({
                "cutoff_pd": c, "approval_rate": 0.0, "approved_bad_rate": np.nan,
                "expected_profit": 0.0, "realized_profit": 0.0,
                "expected_loss": 0.0,
            })
            continue
        realized = (
            np.sum((1 - y_true[approved]) * margin_rate * ead)
            - np.sum(y_true[approved] * lgd * ead)
        )
        expected = float(np.sum(
            (1 - y_prob[approved]) * margin_rate * ead
            - y_prob[approved] * lgd * ead
        ))
        rows.append({
            "cutoff_pd": c,
            "approval_rate": n_app / len(y_true),
            "approved_bad_rate": float(y_true[approved].mean()),
            "expected_profit": expected,
            "realized_profit": float(realized),
            "expected_loss": float(np.sum(expected_loss(y_prob[approved], lgd, ead))),
        })
    return pd.DataFrame(rows)


def breakeven_pd(*, lgd: float = config.LGD,
                 margin_rate: float = config.MARGIN_RATE) -> float:
    """The PD at which expected profit flips sign.

    Solve ``(1 - PD) * margin - PD * lgd = 0`` for PD.
    """
    return float(margin_rate / (margin_rate + lgd))


def optimal_cutoff(y_true, y_prob, *, lgd: float = config.LGD,
                   margin_rate: float = config.MARGIN_RATE,
                   ead: float = 1.0) -> float:
    """Empirically pick the cut-off with the highest realized profit."""
    table = cutoff_table(y_true, y_prob, lgd=lgd, margin_rate=margin_rate, ead=ead)
    best = table.loc[table["realized_profit"].idxmax()]
    return float(best["cutoff_pd"])


def profit_at_cutoff(y_true, y_prob, cutoff: float, *, lgd: float = config.LGD,
                     margin_rate: float = config.MARGIN_RATE,
                     ead: float = 1.0) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    approved = y_prob < cutoff
    n_app = int(approved.sum())
    if n_app == 0:
        return {"cutoff_pd": cutoff, "approval_rate": 0.0,
                "approved_bad_rate": np.nan, "realized_profit": 0.0}
    realized = (
        np.sum((1 - y_true[approved]) * margin_rate * ead)
        - np.sum(y_true[approved] * lgd * ead)
    )
    return {
        "cutoff_pd": cutoff,
        "approval_rate": n_app / len(y_true),
        "approved_bad_rate": float(y_true[approved].mean()),
        "realized_profit": float(realized),
    }
