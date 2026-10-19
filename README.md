# Credit Risk Lab

An end-to-end credit-default modelling pipeline you can read, run, and learn from.
It builds the two models a real credit risk team runs side by side — a **logistic
scorecard** and a **gradient-boosted challenger** — validates them the way a risk
analyst actually has to, and turns the output into a decision with reason codes.

If you are learning credit risk as a data analyst, this repo is the project that
maps 1:1 onto the job: binning, WoE/IV, scorecard scaling, KS/Gini/PSI, calibration,
cut-offs and expected loss.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

That trains the models (about a minute) and opens the dashboard at
`http://127.0.0.1:8000`, where you can score an applicant, see the PD, the points
score, the approve/decline decision and the adverse-action reason codes.

```bash
python run.py --retrain          # rebuild the models from scratch
python run.py --csv my_data.csv  # train on a real dataset instead of synthetic
python run.py --no-browser       # don't pop a browser window
python -m src.pipeline           # just run the pipeline, no web server
```

---

## What it actually does

```
data  ->  WoE binning  ->  scorecard  ->  challenger  ->  validation  ->  decision
```

1. **Data** (`src/data.py`) — generates a synthetic retail-credit book of 25,000
   applicants with a realistic latent default process, plus a *drifted* out-of-time
   sample. You can swap in a real CSV instead.
2. **WoE binning** (`src/features.py`) — bins every variable and replaces it with
   its Weight of Evidence; computes Information Value per feature.
3. **Scorecard** (`src/scorecard.py`) — logistic regression on the WoE values,
   scaled to points (`+40 points doubles the odds of being good`).
4. **Challenger** (`src/challenger.py`) — gradient boosting, with permutation
   importance and SHAP if it is installed.
5. **Validation** (`src/validation.py`) — AUC/Gini, KS, PSI, calibration, lift/gains.
6. **Decision** (`src/decision.py`) — PD↔score conversion, break-even PD,
   cut-off sweep, expected loss and profit.

---

## The concepts, in the order you hit them

**PD / LGD / EAD.** Probability of Default, Loss Given Default, Exposure At Default.
Expected loss = `PD × LGD × EAD`. Almost everything in credit risk is a variation on
that one line.

**Weight of Evidence (WoE).** For each bin of a variable:

```
WoE = ln( %Good in bin / %Bad in bin )
```

Binning + WoE (a) linearises the feature against the log-odds of default, which is
exactly what logistic regression wants; (b) makes missing values and outliers their
own bin; (c) keeps the model monotonic and explainable — a regulator can read it.

**Information Value (IV).** The strength of a variable:
IV < 0.02 useless · 0.02–0.1 weak · 0.1–0.3 medium · 0.3–0.5 strong · > 0.5 suspicious.
Look at `artifacts/reports/iv_summary.csv` after a run.

**Scorecard scaling.** With points-to-double-the-odds (`PDO`):

```
factor = PDO / ln(2)
offset = base_score - factor × ln(base_odds)
score  = offset - factor × logit(default)
```

The default here is `base_score=600`, `base_odds=20:1`, `PDO=40`. Each bin of each
variable contributes a fixed number of points, so the whole model prints as a table
(`artifacts/reports/scorecard_points.csv`).

**The metrics that matter** (accuracy is worthless at a 15% bad rate):

- **AUC / Gini** — ranking power. `Gini = 2×AUC − 1`.
- **KS** — the maximum vertical gap between the good and bad score distributions.
- **PSI** — population stability between train and out-of-time scores.
  < 0.10 stable · 0.10–0.25 monitor · > 0.25 recalibrate.
- **Calibration** — do predicted PDs match observed default rates?
- **Lift / gains** — how much better than random the model is in the risky deciles.

**The decision.** A PD is not a decision. Approve while expected profit is positive:

```
E[profit] = (1 - PD) × margin_rate − PD × LGD
```

which gives the break-even PD `margin / (margin + LGD)`. That is risk-based pricing
and cut-off setting in one line.

---

## Results from the built-in synthetic book

| model | Gini (OOT) | KS (OOT) |
|---|---|---|
| Scorecard | 0.667 | 0.528 |
| Challenger | **0.682** | **0.544** |

- Train bad rate 15.5%, out-of-time bad rate 23.4% (deliberate drift).
- Score PSI 0.145 → **"monitor"**: the population moved, a risk team would
  investigate before trusting the model on new business.
- Break-even PD 15.6%; the profit-maximising cut-off lands near it.

The challenger wins because the synthetic data contains **interactions** that a
per-variable WoE scorecard structurally cannot represent. That is the normal reason
a challenger exists. Note how *small* the gap is — when risk is close to additive a
well-built scorecard matches a GBM, which is why scorecards are still everywhere.

---

## Outputs

```
artifacts/
  model_bundle.pkl              fitted binner + scorecard + challenger + metadata
  reports/
    report.json                 headline metrics and insights
    binning_table.csv           every bin: counts, bad rate, WoE, IV
    scorecard_points.csv        the full points card
    scorecard_summary.csv       points swing per variable
    iv_summary.csv              Information Value league table
    validation_valid.csv        in-time hold-out metrics
    validation_oot.csv          out-of-time metrics
    calibration_oot.csv         predicted vs observed PD
    deciles_oot.csv             decile bad rates and lift
    cutoff_oot.csv              approval rate / bad rate / profit by cut-off
  figures/                      10 validation charts
data/raw/sample_portfolio.csv   a look at the raw data
```

---

## Using a real dataset

Any CSV with a binary target works:

```bash
python run.py --csv data/raw/lending_club.csv --retrain
```

The loader auto-detects common target names (`default`, `loan_status`, `TARGET`,
`SeriousDlqin2yrs`, …) and infers which columns are numeric vs categorical. Public
datasets worth trying: **German Credit** (small, classic), **Lending Club**
(real, messy), **Home Credit Default Risk** (the standard Kaggle benchmark).

---

## Suggested learning path

1. Run `python -m src.pipeline`, then open `binning_table.csv` and read it top to
   bottom. Understand why each bin got its WoE.
2. Open `scorecard_points.csv`. Pick one applicant and add up their points by hand;
   check it matches the score the dashboard gives you.
3. Read `iv_summary.csv` and decide which variables you would drop.
4. Look at the `psi`, `calibration` and `gains` charts. Explain the PSI value.
5. Change `BASE_ODDS` or `PDO` in `src/config.py`, rerun, and confirm the Gini and
   KS do **not** change (scaling must not affect ranking).
6. Change `LGD` or `MARGIN_RATE` and watch the optimal cut-off move.
7. **Best exercise:** add a genuinely predictive variable to `src/data.py` and drive
   it through the whole pipeline.

---

## Notes

- Everything is dependency-light: scikit-learn only, no XGBoost/LightGBM required.
  If you `pip install xgboost shap`, the challenger can use SHAP for explainability
  automatically.
- The dashboard is plain HTML/CSS/JS served by FastAPI — no build step.
- Re-running is deterministic (`RANDOM_SEED = 42`), so your metrics will match the
  table above.
