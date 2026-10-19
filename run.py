"""Credit Risk Lab - one command to train the models and open the dashboard.

    python run.py                # train if needed, then serve
    python run.py --retrain      # rebuild the models, then serve
    python run.py --csv my.csv   # train on a real dataset instead of synthetic
    python run.py --no-browser   # don't pop the browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit Risk Lab")
    parser.add_argument("--retrain", action="store_true",
                        help="force a full pipeline re-run")
    parser.add_argument("--csv", default=None,
                        help="path to a real credit dataset (CSV)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.retrain or args.csv or not config.BUNDLE_PATH.exists():
        run_pipeline(csv_path=args.csv)
    else:
        print(f"Using existing model bundle: {config.BUNDLE_PATH}")
        print("(pass --retrain to rebuild it)")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print(f"\nDashboard ready at {url}\n")

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
