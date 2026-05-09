from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from drift_utils import generate_drift_report, save_json


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_PATH = BASE_DIR / "model" / "drift_baseline.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "model" / "drift_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data drift monitoring for FraudGuard AI.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the new transaction CSV batch to compare against the saved baseline.",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE_PATH),
        help="Path to the saved drift baseline JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save the generated drift report JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input batch not found: {input_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Drift baseline not found: {baseline_path}. Run `python train_model.py` first."
        )

    current_batch = pd.read_csv(input_path)
    with baseline_path.open("r", encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)

    report = generate_drift_report(current_batch, baseline)
    report["input_rows"] = int(len(current_batch))
    save_json(output_path, report)

    print(f"Overall drift status: {report['overall_status']}")
    print(f"Saved drift report to: {output_path}")


if __name__ == "__main__":
    main()
