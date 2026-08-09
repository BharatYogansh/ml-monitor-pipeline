"""
FastAPI serving layer.

Loads the trained model once at startup, exposes /predict, and logs every
request's inputs, prediction, and latency to SQLite so the drift detector
and dashboard have something to read.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import db

FEATURE_COLUMNS = [
    "lotsize", "bedrooms", "bathrms", "stories",
    "driveway", "recroom", "fullbase", "gashw", "airco", "garagepl", "prefarea",
]
MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = joblib.load("model.pkl")
    db.init_db()
    yield


app = FastAPI(title="Windsor Housing Price Predictor with Live Monitoring", lifespan=lifespan)


class HouseFeatures(BaseModel):
    lotsize: float = Field(..., gt=0, description="Lot size, sq ft")
    bedrooms: int = Field(..., ge=1, le=10)
    bathrms: int = Field(..., ge=1, le=5)
    stories: int = Field(..., ge=1, le=4)
    driveway: int = Field(..., ge=0, le=1, description="1 = has driveway")
    recroom: int = Field(..., ge=0, le=1, description="1 = has recreation room")
    fullbase: int = Field(..., ge=0, le=1, description="1 = has full basement")
    gashw: int = Field(..., ge=0, le=1, description="1 = gas water heating")
    airco: int = Field(..., ge=0, le=1, description="1 = has central air conditioning")
    garagepl: int = Field(..., ge=0, le=4, description="number of garage places")
    prefarea: int = Field(..., ge=0, le=1, description="1 = in preferred neighborhood")


class PredictionResponse(BaseModel):
    prediction: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    start = time.perf_counter()
    X = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
    pred = float(MODEL.predict(X)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    row = features.model_dump()
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["prediction"] = pred
    row["latency_ms"] = latency_ms
    db.insert_prediction(row)

    return PredictionResponse(prediction=pred, latency_ms=latency_ms)


@app.get("/stats")
def stats():
    df = db.fetch_all()
    if df.empty:
        return {"total_predictions": 0, "avg_latency_ms": None, "p95_latency_ms": None}
    return {
        "total_predictions": int(len(df)),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
    }
