"""End-to-end pipeline: data -> WoE -> scorecard -> challenger -> validate -> decide.

Running this produces every artifact the scoring app and the report need:

    artifacts/model_bundle.pkl        fitted binner + scorecard + challenger + meta
    artifacts/reports/report.json     headline metrics and insights
    artifacts/reports/*.csv           binning tables, card, validation tables
    artifacts/figures/*.png           validation charts
    data/raw/sample_portfolio.csv     a peek at the data
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from . import config, decision, plots, validation
from . import data as data_mod
from .challenger import Challenger
from .features import WoEBinner
from .scorecard import Scorecard


# --------------------------------------------------------------------- helpers
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def infer_features(df: pd.DataFrame, target: str) -> list[str]:
    """Pick modelling columns from an arbitrary frame (used for real CSVs)."""
    cols = []
    for col in df.columns:
        if col == target or col.lower() in {"customer_id", "id", "index"}:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(col)
        elif s.nunique(dropna=True) <= 30:
            cols.append(col)
    return cols


def _load_datasets(csv_path: str | None):
    if csv_path:
        df, target = data_mod.load_csv(csv_path)
        features = infer_features(df, target)
        n = len(df)
        idx = np.random.default_rng(config.RANDOM_SEED).permutation(n)
        a, b = int(0.60 * n), int(0.80 * n)
        train = df.iloc[idx[:a]].reset_index(drop=True)
        valid = df.iloc[idx[a:b]].reset_index(drop=True)
        oot = df.iloc[idx[b:]].reset_index(drop=True)
        source = f"csv:{csv_path}"
    else:
        ds = data_mod.build_datasets()
        train, valid, oot = ds["train"], ds["valid"], ds["oot"]
        target = config.TARGET
        features = data_mod.FEATURES
        source = "synthetic"

    # Persist a sample so the data is inspectable.
    pd.concat([train.head(2500), valid.head(1500), oot.head(1500)]).to_csv(
        config.RAW_DIR / "sample_portfolio.csv", index=False
    )
    return train, valid, oot, features, target, source


def _category_options(train: pd.DataFrame, features: list[str]) -> dict:
    options = {}
    for col in features:
        if not pd.api.types.is_numeric_dtype(train[col]):
            options[col] = sorted(train[col].dropna().astype(str).unique().tolist())
    return options


# --------------------------------------------------------------------- pipeline
def run_pipeline(csv_path: str | None = None, verbose: bool = True) -> dict:
    train, valid, oot, features, target, source = _load_datasets(csv_path)

    if verbose:
        print("=" * 68)
        print("CREDIT RISK LAB :: end-to-end pipeline")
        print("=" * 68)
        print(f"source        : {source}")
        print(f"features      : {len(features)} variables")
        print(f"train/valid/oot: {len(train):,} / {len(valid):,} / {len(oot):,}")
        print(f"train bad rate: {train[target].mean():.2%}   "
              f"oot bad rate: {oot[target].mean():.2%}")

    X_train, y_train = train[features], train[target]
    X_valid, y_valid = valid[features], valid[target]
    X_oot, y_oot = oot[features], oot[target]

    # 1) WoE binning ------------------------------------------------------
    binner = WoEBinner(max_bins=8, min_bin_pct=0.03).fit(X_train, y_train)
    if verbose:
        print(f"\n[1/6] WoE binning fitted -> {len(binner.columns_)} variables")

    # 2) Scorecard --------------------------------------------------------
    scorecard = Scorecard(binner).fit(X_train, y_train)
    sc_valid = scorecard.predict_pd(X_valid)
    sc_oot = scorecard.predict_pd(X_oot)
    if verbose:
        print(f"[2/6] Scorecard fitted  (base {config.BASE_SCORE}, "
              f"{config.BASE_ODDS:g}:1 odds, PDO {config.PDO})")

    # 3) Challenger -------------------------------------------------------
    challenger = Challenger().fit(X_train, y_train)
    ch_valid = challenger.predict_pd(X_valid)
    ch_oot = challenger.predict_pd(X_oot)
    if verbose:
        print("[3/6] Challenger (gradient boosting) fitted")

    # 4) Validation -------------------------------------------------------
    valid_cmp = validation.comparison_table(
        {"scorecard": sc_valid, "challenger": ch_valid}, y_valid
    )
    oot_cmp = validation.comparison_table(
        {"scorecard": sc_oot, "challenger": ch_oot}, y_oot
    )
    scorecard_metrics = oot_cmp[oot_cmp["model"] == "scorecard"].iloc[0].to_dict()
    challenger_metrics = oot_cmp[oot_cmp["model"] == "challenger"].iloc[0].to_dict()

    ks_curve = validation.ks_curve(y_oot, sc_oot)
    train_scores = scorecard.predict_score(X_train)
    oot_scores = scorecard.predict_score(X_oot)
    psi_res = validation.psi(train_scores, oot_scores)
    calibration = validation.calibration_table(y_oot, ch_oot)
    deciles = validation.decile_table(y_oot, ch_oot)
    if verbose:
        print(f"[4/6] Validation  scorecard Gini {scorecard_metrics['gini']:.3f} "
              f"/ KS {scorecard_metrics['ks']:.3f}   "
              f"challenger Gini {challenger_metrics['gini']:.3f} "
              f"/ KS {challenger_metrics['ks']:.3f}")
        print(f"      Score PSI (train vs oot) = {psi_res['psi']:.3f} "
              f"[{psi_res['band']}]")

    # 5) Decision / economics --------------------------------------------
    breakeven = decision.breakeven_pd()
    tradeoff = decision.cutoff_table(y_oot, ch_oot)
    opt_cutoff = decision.optimal_cutoff(y_oot, ch_oot)
    if verbose:
        print(f"[5/6] Break-even PD = {breakeven:.3f}  |  "
              f"profit-max cut-off = {opt_cutoff:.3f}")

    # 6) Figures ----------------------------------------------------------
    figures = {}
    importance = challenger.feature_importance(X_valid, y_valid)
    shap_df = challenger.shap_values(X_valid)
    figures["score_distribution"] = plots.plot_score_distribution(oot_scores, y_oot)
    figures["roc"] = plots.plot_roc(y_oot, {"scorecard": sc_oot, "challenger": ch_oot})
    figures["ks"] = plots.plot_ks(ks_curve)
    figures["gains"] = plots.plot_gains(y_oot, {"scorecard": sc_oot, "challenger": ch_oot})
    figures["calibration"] = plots.plot_calibration(y_oot, {"challenger": ch_oot})
    figures["psi"] = plots.plot_psi(psi_res["table"], psi_res["psi"])
    figures["iv"] = plots.plot_importance(
        binner.iv_summary(), "iv", "Information Value by feature", "iv.png"
    )
    figures["importance"] = plots.plot_importance(
        importance, "importance", "Challenger permutation importance",
        "importance.png"
    )
    figures["cutoff"] = plots.plot_cutoff(tradeoff, breakeven)
    figures["profit"] = plots.plot_profit(tradeoff)
    if verbose:
        print(f"[6/6] Figures written -> {config.FIGURE_DIR}")

    # ---- persist tables -------------------------------------------------
    binner.binning_table().to_csv(config.REPORT_DIR / "binning_table.csv", index=False)
    scorecard.points_table_.to_csv(config.REPORT_DIR / "scorecard_points.csv", index=False)
    scorecard.card_summary().to_csv(config.REPORT_DIR / "scorecard_summary.csv", index=False)
    binner.iv_summary().to_csv(config.REPORT_DIR / "iv_summary.csv", index=False)
    valid_cmp.to_csv(config.REPORT_DIR / "validation_valid.csv", index=False)
    oot_cmp.to_csv(config.REPORT_DIR / "validation_oot.csv", index=False)
    calibration.to_csv(config.REPORT_DIR / "calibration_oot.csv", index=False)
    deciles.to_csv(config.REPORT_DIR / "deciles_oot.csv", index=False)
    tradeoff.to_csv(config.REPORT_DIR / "cutoff_oot.csv", index=False)
    psi_res["table"].to_csv(config.REPORT_DIR / "psi_table.csv", index=False)
    importance.to_csv(config.REPORT_DIR / "challenger_importance.csv", index=False)

    # ---- meta + bundle --------------------------------------------------
    meta = {
        "source": source,
        "features": features,
        "target": target,
        "numeric_features": [c for c in features if pd.api.types.is_numeric_dtype(train[c])],
        "categorical_features": [c for c in features if not pd.api.types.is_numeric_dtype(train[c])],
        "category_options": _category_options(train, features),
        "base_score": config.BASE_SCORE,
        "base_odds": config.BASE_ODDS,
        "pdo": config.PDO,
        "lgd": config.LGD,
        "margin_rate": config.MARGIN_RATE,
        "default_cutoff_pd": config.DEFAULT_CUTOFF_PD,
        "breakeven_pd": round(breakeven, 4),
        "optimal_cutoff_pd": round(opt_cutoff, 4),
        "train_bad_rate": round(float(y_train.mean()), 4),
        "oot_bad_rate": round(float(y_oot.mean()), 4),
        "psi": psi_res["psi"],
        "scorecard_metrics": scorecard_metrics,
        "challenger_metrics": challenger_metrics,
    }
    bundle = {
        "binner": binner,
        "scorecard": scorecard,
        "challenger": challenger,
        "meta": meta,
        "examples": valid[features].head(250).reset_index(drop=True),
    }
    with open(config.BUNDLE_PATH, "wb") as fh:
        pickle.dump(bundle, fh)

    report = {
        "meta": meta,
        "scorecard_valid": valid_cmp[valid_cmp["model"] == "scorecard"].iloc[0].to_dict(),
        "challenger_valid": valid_cmp[valid_cmp["model"] == "challenger"].iloc[0].to_dict(),
        "scorecard_oot": scorecard_metrics,
        "challenger_oot": challenger_metrics,
        "psi": {"value": psi_res["psi"], "band": psi_res["band"]},
        "calibration": _jsonable(calibration.to_dict(orient="records")),
        "deciles": _jsonable(deciles.to_dict(orient="records")),
        "cutoff": _jsonable(tradeoff.to_dict(orient="records")),
        "top_iv": _jsonable(binner.iv_summary().head(10).to_dict(orient="records")),
        "top_importance": _jsonable(importance.head(10).to_dict(orient="records")),
        "top_shap": _jsonable(shap_df.head(10).to_dict(orient="records")) if shap_df is not None else None,
        "card_summary": _jsonable(scorecard.card_summary().to_dict(orient="records")),
        "figures": {k: str(v) for k, v in figures.items()},
    }
    with open(config.REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2)

    if verbose:
        print("\nArtifacts:")
        print(f"  bundle  : {config.BUNDLE_PATH}")
        print(f"  report  : {config.REPORT_PATH}")
        print(f"  figures : {len(figures)} charts")
        print("=" * 68)

    return {"meta": meta, "report": report, "bundle_path": str(config.BUNDLE_PATH)}


def load_bundle() -> dict:
    with open(config.BUNDLE_PATH, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    run_pipeline()
