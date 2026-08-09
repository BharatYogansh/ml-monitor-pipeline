"""
Sends synthetic traffic to a running instance of the API, built from real
Windsor Housing rows:
  1. A batch of "normal" requests: bootstrap-sampled real rows.
  2. A batch of "drifted" requests: real rows perturbed toward larger,
     higher-amenity homes (simulated market shift).

This exists so you can demo and screenshot the drift alert on demand,
without waiting for real traffic patterns to change.

Usage:
    python simulate_traffic.py --url http://localhost:8000 --normal 60 --drift 60
"""
import argparse
import time

import requests

from data_gen import FEATURE_COLUMNS, load_real_data, sample_drifted, sample_normal


def send_batch(url: str, df, label: str, delay: float):
    ok, failed = 0, 0
    for _, row in df.iterrows():
        payload = {col: float(row[col]) for col in FEATURE_COLUMNS}
        try:
            r = requests.post(f"{url}/predict", json=payload, timeout=5)
            r.raise_for_status()
            ok += 1
        except requests.RequestException as e:
            failed += 1
            print(f"  request failed: {e}")
        time.sleep(delay)
    print(f"[{label}] sent {ok} ok, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=int, default=60)
    parser.add_argument("--drift", type=int, default=60)
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()

    reference = load_real_data()

    print("Sending normal traffic...")
    normal_df = sample_normal(reference, args.normal)
    send_batch(args.url, normal_df, "normal", args.delay)

    if args.drift > 0:
        print("Sending drifted traffic...")
        drift_df = sample_drifted(reference, args.drift)
        send_batch(args.url, drift_df, "drift", args.delay)


if __name__ == "__main__":
    main()
