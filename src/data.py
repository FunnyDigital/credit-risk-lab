"""Data layer.

Two ways in:

1. ``generate_portfolio`` - a synthetic but realistic retail-credit book. Every
   feature has a sensible relationship to the probability of default, so the
   models you build actually learn something interpretable.
2. ``load_csv`` - point it at a real file (Lending Club, German Credit, Home
   Credit, ...) and it will hand back a frame the pipeline can consume.

The synthetic generator also produces a *drifted* out-of-time sample, which is
what lets us demonstrate PSI (population stability) honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "installment",
    "dti",
    "fico_score",
    "revolving_utilization",
    "delinquencies_2yrs",
    "inquiries_6m",
    "num_open_accounts",
    "credit_history_months",
    "loan_to_income",
    "months_since_last_delinquency",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
    "grade",
    "income_verified",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_portfolio(
    n: int = config.N_APPLICANTS,
    seed: int = config.RANDOM_SEED,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic retail-credit portfolio.

    Parameters
    ----------
    n : number of applicants
    seed : RNG seed
    drift : >=0. A positive value shifts key distributions (lower bureau score,
            higher utilisation and DTI, fewer verified incomes) to emulate an
            out-of-time or stressed book. The default behaviour is still driven
            by the same latent risk function, so the OOT sample genuinely is
            riskier.
    """
    rng = np.random.default_rng(seed)

    # --- borrower demographics -------------------------------------------
    age = np.clip(rng.normal(41, 12, n), 18, 80).round().astype(int)
    employment_years = np.clip(rng.gamma(2.0, 3.5, n) - 0.4 * drift, 0, age - 18)
    annual_income = np.clip(
        rng.lognormal(mean=10.85 - 0.02 * drift, sigma=0.55, size=n), 12_000, 500_000
    )

    # --- credit bureau characteristics -----------------------------------
    fico_score = np.clip(rng.normal(697 - 38 * drift, 72, n), 300, 850)
    revolving_utilization = np.clip(rng.gamma(2.2, 0.22, n) + 0.12 * drift, 0, 1.7)
    delinquencies_2yrs = rng.poisson(0.35 + 0.20 * drift, n)
    inquiries_6m = rng.poisson(1.1 + 0.35 * drift, n)
    num_open_accounts = np.clip(rng.poisson(9, n), 0, 40)
    credit_history_months = np.clip(rng.normal(188, 90, n), 6, 480).round().astype(int)
    dti = np.clip(rng.normal(18 + 3.5 * drift, 9, n), 0, 65)

    # --- categoricals -----------------------------------------------------
    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"], n, p=[0.42, 0.44, 0.14]
    )
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "medical",
         "small_business"],
        n,
        p=[0.34, 0.24, 0.12, 0.14, 0.08, 0.08],
    )
    income_verified = rng.random(n) < (0.44 - 0.10 * drift)

    # --- loan terms -------------------------------------------------------
    loan_term_months = rng.choice([36, 60], n, p=[0.62, 0.38])
    loan_amount = np.clip(
        annual_income * rng.uniform(0.05, 0.60, n), 1_000, 60_000
    ).round()

    interest_rate = np.clip(
        6.0
        + (750 - fico_score) * 0.090
        + (loan_term_months - 36) * 0.020
        + revolving_utilization * 3.0
        + rng.normal(0, 1.1, n),
        3.9,
        34.0,
    ).round(2)

    monthly_rate = interest_rate / 1200.0
    installment = np.where(
        monthly_rate > 0,
        loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-loan_term_months)),
        loan_amount / loan_term_months,
    ).round(2)

    loan_to_income = (loan_amount / annual_income).round(4)

    grade = pd.cut(
        fico_score,
        bins=[300, 580, 620, 660, 700, 740, 780, 851],
        labels=["G", "F", "E", "D", "C", "B", "A"],
        right=False,
    ).astype(str)

    months_since_last_delinquency = np.where(
        delinquencies_2yrs > 0,
        np.clip(rng.exponential(14, n), 1, 72).round(),
        np.nan,
    )

    # --- latent default process ------------------------------------------
    # Signs are chosen so the model has a realistic, learnable structure.
    # Note the interaction terms at the end: a logistic scorecard built on
    # per-variable WoE bins cannot represent these, which is exactly the edge
    # a gradient-boosted challenger is supposed to exploit.
    z = (
        -3.55
        + 0.0260 * (650.0 - fico_score) / 10.0
        + 1.15 * revolving_utilization
        + 0.34 * delinquencies_2yrs
        + 0.065 * inquiries_6m
        + 0.040 * (dti - 18)
        + 2.10 * loan_to_income
        + 0.055 * (interest_rate - 12)
        - 0.95 * income_verified
        - 0.030 * employment_years
        + 0.40 * (home_ownership == "RENT")
        - 0.30 * (home_ownership == "OWN")
        + 0.55 * (purpose == "small_business")
        + 0.22 * (purpose == "medical")
        - 0.18 * (purpose == "credit_card")
        # interactions
        + 1.80 * ((revolving_utilization > 0.65) & (delinquencies_2yrs > 0))
        + 1.30 * ((loan_to_income > 0.45) & (fico_score < 660))
        + 1.00 * ((dti > 32) & (~income_verified))
        + 1.50 * ((fico_score < 620) & (revolving_utilization > 0.5))
    )
    pd_true = _sigmoid(z)
    default = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{seed:02d}{i:07d}" for i in range(n)],
            "age": age,
            "annual_income": annual_income.round(0),
            "employment_years": employment_years.round(1),
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "installment": installment,
            "dti": dti.round(2),
            "fico_score": fico_score.round(0).astype(int),
            "revolving_utilization": revolving_utilization.round(4),
            "delinquencies_2yrs": delinquencies_2yrs,
            "inquiries_6m": inquiries_6m,
            "num_open_accounts": num_open_accounts,
            "credit_history_months": credit_history_months,
            "loan_to_income": loan_to_income,
            "months_since_last_delinquency": months_since_last_delinquency,
            "home_ownership": pd.Series(home_ownership, dtype="object"),
            "purpose": pd.Series(purpose, dtype="object"),
            "grade": pd.Series(grade, dtype="object"),
            "income_verified": np.where(income_verified, "Yes", "No"),
            "pd_true": pd_true.round(6),
            config.TARGET: default,
        }
    )
    return df


def load_csv(path: str, target: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load a real credit dataset.

    Returns the frame plus the name of the binary target column. If ``target``
    is not supplied we try a few common names.
    """
    df = pd.read_csv(path)
    if target is None:
        for candidate in ("default", "Default", "loan_status", "TARGET",
                          "SeriousDlqin2yrs", "class", "credit_risk"):
            if candidate in df.columns:
                target = candidate
                break
    if target is None or target not in df.columns:
        raise ValueError(
            f"Could not find a target column in {path}. Pass target=<column>."
        )
    return df, target


def split_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / out-of-time (60/20/20).

    Validation is an in-time hold-out; the OOT set is generated separately in
    the pipeline so that it carries distribution drift.
    """
    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(idx)
    train_end = int(0.60 * n)
    valid_end = int(0.80 * n)
    train = df.iloc[idx[:train_end]].reset_index(drop=True)
    valid = df.iloc[idx[train_end:valid_end]].reset_index(drop=True)
    return train, valid


def build_datasets() -> dict[str, pd.DataFrame]:
    """Convenience: full train/valid/oot dictionary from the synthetic book."""
    full = generate_portfolio()
    train, valid = split_portfolio(full)
    oot = generate_portfolio(
        n=int(config.N_APPLICANTS * 0.25),
        seed=config.RANDOM_SEED + 999,
        drift=config.OOT_DRIFT,
    )
    return {"train": train, "valid": valid, "oot": oot}
