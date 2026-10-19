"""Central configuration: paths, seeds and scorecard scaling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIGURE_DIR = ARTIFACT_DIR / "figures"

for _d in (RAW_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_APPLICANTS = 25_000
TARGET = "default"

# How the out-of-time (OOT) sample drifts away from the training sample.
# This is what makes the PSI / stability analysis meaningful.
OOT_DRIFT = 0.55

# ---------------------------------------------------------------------------
# Scorecard scaling
# ---------------------------------------------------------------------------
# Points-to-Double-the-Odds scaling:
#   +PDO points  <->  odds of being good double.
BASE_SCORE = 600
BASE_ODDS = 20.0          # 20:1 good:bad at BASE_SCORE
PDO = 40

# ---------------------------------------------------------------------------
# Economics of a loan (used for cut-off / expected-loss analysis)
# ---------------------------------------------------------------------------
LGD = 0.65                # loss given default (fraction of exposure lost)
MARGIN_RATE = 0.12        # interest margin earned on a performing loan
DEFAULT_CUTOFF_PD = 0.20  # naive cut-off used when economics are ignored

BUNDLE_PATH = ARTIFACT_DIR / "model_bundle.pkl"
REPORT_PATH = REPORT_DIR / "report.json"
