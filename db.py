"""
Tiny SQLite helper used by the API (to log predictions) and by the drift
detector / dashboard (to read them back). Kept dependency-free on purpose —
this is a demo project, not a claim that SQLite belongs in real production.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

DB_PATH = "logs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    size_sqft REAL,
    bedrooms INTEGER,
    age_years REAL,
    location_score REAL,
    distance_to_city_km REAL,
    prediction REAL,
    latency_ms REAL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def insert_prediction(row: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO predictions
            (timestamp, size_sqft, bedrooms, age_years, location_score, distance_to_city_km, prediction, latency_ms)
            VALUES (:timestamp, :size_sqft, :bedrooms, :age_years, :location_score, :distance_to_city_km, :prediction, :latency_ms)""",
            row,
        )
        conn.commit()


def fetch_recent(n: int = 200) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(n,))
    return df.iloc[::-1].reset_index(drop=True)


def fetch_all() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)
