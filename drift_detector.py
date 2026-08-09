"""
Compares recently logged prediction inputs against the reference training
distribution, per feature, using the two-sample Kolmogorov-Smirnov test.

A low p-value (< ALPHA) means the recent feature values are statistically
unlikely to come from the same distribution as the training data — i.e.
the input data has drifted, and the model's predictions can no longer be
trusted the way they were at training time.

Run standalone (python drift_detector.py) or call run_drift_check() from
the dashboard / a scheduler.
"""
import json
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp

import db

FEATURE_COLUMNS = ["size_sqft", "bedrooms", "age_years", "location_score", "distance_to_city_km"]
REFERENCE_PATH = "reference_data.csv"
REPORT_PATH = "drift_report.json"
ALPHA = 0.05
RECENT_WINDOW = 100


def run_drift_check(recent_window: int = RECENT_WINDOW) -> dict:
    reference = pd.read_csv(REFERENCE_PATH)
    recent = db.fetch_recent(recent_window)

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": int(len(recent)),
        "drift_detected": False,
        "drifted_columns": [],
        "column_details": {},
    }

    if len(recent) < 20:
        report["note"] = f"Need at least 20 recent predictions to test for drift (have {len(recent)})."
        _save(report)
        return report

    for col in FEATURE_COLUMNS:
        stat, p_value = ks_2samp(reference[col], recent[col])
        drifted = p_value < ALPHA
        report["column_details"][col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 5),
            "drifted": bool(drifted),
            "reference_mean": round(float(reference[col].mean()), 2),
            "recent_mean": round(float(recent[col].mean()), 2),
        }
        if drifted:
            report["drifted_columns"].append(col)

    report["drift_detected"] = len(report["drifted_columns"]) > 0
    _save(report)
    return report


def _save(report: dict):
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    result = run_drift_check()
    status = "DRIFT DETECTED" if result["drift_detected"] else "HEALTHY"
    print(f"[{status}] sample_size={result['sample_size']} drifted_columns={result.get('drifted_columns')}")
