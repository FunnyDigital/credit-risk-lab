"""FastAPI scoring service + dashboard.

Endpoints
---------
GET  /                the dashboard
GET  /api/meta        feature schema, model metrics and figure URLs
GET  /api/example     a random applicant drawn from the validation sample
POST /api/score       score one applicant (PD, score, decision, reason codes)
GET  /figures/*.png   the validation charts produced by the pipeline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src import config, decision  # noqa: E402
from src.pipeline import load_bundle  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Credit Risk Lab", version="1.0.0")

_bundle: dict | None = None


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not config.BUNDLE_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model found. Run `python run.py --retrain` first.",
            )
        _bundle = load_bundle()
    return _bundle


def _risk_band(pd_value: float, breakeven: float) -> str:
    if pd_value < 0.05:
        return "Low"
    if pd_value < 0.10:
        return "Moderate"
    if pd_value < breakeven:
        return "Elevated"
    if pd_value < 0.30:
        return "High"
    return "Very high"


def _reason_count(payload: dict) -> int:
    try:
        return max(1, min(int(payload.get("top_reasons", 6)), 12))
    except (TypeError, ValueError):
        return 6


def _to_row(payload: dict, meta: dict) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for feat in meta["features"]:
        value = payload.get(feat)
        if feat in meta["numeric_features"]:
            if value in (None, ""):
                row[feat] = float("nan")
            else:
                try:
                    row[feat] = float(value)
                except (TypeError, ValueError):
                    row[feat] = float("nan")
        else:
            row[feat] = "MISSING" if value in (None, "") else str(value)
    return pd.DataFrame([row], columns=meta["features"])


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta() -> dict:
    bundle = get_bundle()
    info = bundle["meta"]
    figures = {}
    if config.REPORT_PATH.exists():
        report = json.loads(config.REPORT_PATH.read_text(encoding="utf-8"))
        figures = {k: f"/figures/{Path(v).name}" for k, v in report.get("figures", {}).items()}
        info = {**info, "report": {
            "top_iv": report.get("top_iv", []),
            "card_summary": report.get("card_summary", []),
        }}
    return {"meta": info, "figures": figures}


@app.get("/api/example")
def example() -> dict:
    bundle = get_bundle()
    examples = bundle["examples"]
    row = examples.sample(n=1, random_state=None).iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


@app.post("/api/score")
def score(payload: dict) -> dict:
    bundle = get_bundle()
    meta = bundle["meta"]
    frame = _to_row(payload, meta)

    pd_sc = float(bundle["scorecard"].predict_pd(frame)[0])
    score_value = float(bundle["scorecard"].predict_score(frame)[0])
    pd_ch = float(bundle["challenger"].predict_pd(frame)[0])
    reasons = bundle["scorecard"].explain(frame, top=_reason_count(payload))

    breakeven = meta["breakeven_pd"]
    lgd = meta["lgd"]
    margin = meta["margin_rate"]
    ead = payload.get("loan_amount")
    try:
        ead_value = float(ead)
    except (TypeError, ValueError):
        ead_value = 1.0

    return {
        "score": round(score_value, 1),
        "score_implied_pd": round(float(decision.score_to_pd(score_value)), 4),
        "scorecard_pd": round(pd_sc, 4),
        "challenger_pd": round(pd_ch, 4),
        "decision": "APPROVE" if pd_sc < breakeven else "DECLINE",
        "risk_band": _risk_band(pd_sc, breakeven),
        "breakeven_pd": breakeven,
        "optimal_cutoff_pd": meta["optimal_cutoff_pd"],
        "expected_loss": round(pd_sc * lgd * ead_value, 2),
        "expected_profit": round((1 - pd_sc) * margin * ead_value - pd_sc * lgd * ead_value, 2),
        "reason_codes": reasons,
    }


app.mount("/figures", StaticFiles(directory=str(config.FIGURE_DIR)), name="figures")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
