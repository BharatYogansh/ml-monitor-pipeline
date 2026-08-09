"""
Synthetic housing-price data generator.

Used two ways in this project:
  - train_model.py calls this with drift=False to build the training set
    and the "reference" distribution the drift detector compares against.
  - simulate_traffic.py calls this with drift=True to generate requests
    that look like a shifted market (e.g. a new, much pricier region),
    so you can trigger and see a drift alert on demand.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["size_sqft", "bedrooms", "age_years", "location_score", "distance_to_city_km"]


def generate_housing_data(n: int = 2000, drift: bool = False, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if not drift:
        size_sqft = rng.normal(1500, 400, n).clip(300, 4000)
        bedrooms = rng.integers(1, 6, n)
        age_years = rng.uniform(0, 50, n)
        location_score = rng.uniform(1, 10, n)
        distance_to_city_km = rng.exponential(8, n).clip(0.1, 60)
    else:
        # Deliberately shifted distributions (bigger, newer, pricier, closer-in
        # properties) to simulate a real-world data/market drift scenario.
        size_sqft = rng.normal(4500, 900, n).clip(1000, 9000)
        bedrooms = rng.integers(4, 10, n)
        age_years = rng.uniform(0, 5, n)
        location_score = rng.uniform(6, 10, n)
        distance_to_city_km = rng.exponential(2, n).clip(0.1, 15)

    noise = rng.normal(0, 15000, n)
    price = (
        size_sqft * 120
        + bedrooms * 8000
        - age_years * 900
        + location_score * 9000
        - distance_to_city_km * 1500
        + 20000
        + noise
    ).clip(20000, None)

    return pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "location_score": location_score,
            "distance_to_city_km": distance_to_city_km,
            "price": price,
        }
    )
