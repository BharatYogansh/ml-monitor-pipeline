"""
Real housing data: the Windsor Housing Price dataset (Anglin & Gencay, 1996) -
546 real home sales in Windsor, Ontario, a classic hedonic-pricing dataset
used throughout econometrics teaching (bundled in the `pydataset` package,
resolved locally - no network fetch needed at runtime).

- load_real_data(): the full real dataset, used for training + as the drift
  reference distribution.
- sample_normal(): bootstrap-samples real rows, for "normal" simulated traffic.
- sample_drifted(): bootstrap-samples real rows then perturbs them toward
  larger, higher-amenity homes - simulating a market shift so you can trigger
  a drift alert on demand.
"""
import numpy as np
import pandas as pd
from pydataset import data as _pydataset

FEATURE_COLUMNS = [
    "lotsize", "bedrooms", "bathrms", "stories",
    "driveway", "recroom", "fullbase", "gashw", "airco", "garagepl", "prefarea",
]
TARGET_COLUMN = "price"
BINARY_COLUMNS = ["driveway", "recroom", "fullbase", "gashw", "airco", "prefarea"]


def load_real_data() -> pd.DataFrame:
    df = _pydataset("Housing").reset_index(drop=True)
    for col in BINARY_COLUMNS:
        df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def sample_normal(df: pd.DataFrame, n: int, seed: int | None = None) -> pd.DataFrame:
    return df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def sample_drifted(df: pd.DataFrame, n: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True).copy()
    sample["lotsize"] = (sample["lotsize"] * rng.uniform(1.8, 2.5, n)).astype(int)
    sample["bedrooms"] = (sample["bedrooms"] + rng.integers(1, 3, n)).clip(1, 8)
    sample["bathrms"] = (sample["bathrms"] + rng.integers(1, 2, n)).clip(1, 5)
    sample["garagepl"] = (sample["garagepl"] + rng.integers(1, 2, n)).clip(0, 4)
    sample["airco"] = 1
    sample["prefarea"] = 1
    return sample
