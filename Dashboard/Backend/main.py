# Backend/main.py

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------------------------
# 1) Single FastAPI app + dev CORS (very permissive)
# -------------------------------------------------

app = FastAPI()

# Dev: allow everything so we stop fighting CORS.
# (You can tighten this later.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # <- important
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional tiny debug endpoint
@app.get("/Backend/cors_debug")
def cors_debug():
    return {"message": "CORS is on", "ok": True}

# -------------------------------------------------
# 2) CSV paths + column names
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent          # ...\Dashboard\Backend
ROOT_DIR = BASE_DIR.parent.parent                   # ...\2025-26-Data-driven
DATA_DIR = ROOT_DIR / "Prophet"                     # ...\2025-26-Data-driven\Prophet

PV_CSV = DATA_DIR / "pv_forecast_1-10_comparison_2025_combined.csv"
WIND_CSV = DATA_DIR / "wind_forecast_2025_LGBM_compare.csv"

print(">>> PV CSV:", PV_CSV)
print(">>> WIND CSV:", WIND_CSV)

# ---- column names in your CSVs ----
# adjust these to match exactly pv_df.columns / wind_df.columns

PV_TIME_COL = "Timestamp"
PV_FORECAST_COL = "Forecast_Generation_MWh"
PV_ACTUAL_COL: Optional[str] = "Actual_Generation_MWh"  # or None if no actuals

WIND_TIME_COL = "Timestamp"
WIND_FORECAST_COL = "Forecast_Generation_MWh"
WIND_ACTUAL_COL: Optional[str] = "Real_Generation(MWh)"  # no actuals → leave as None

# -------------------------------------------------
# 3) Load CSVs at startup
# -------------------------------------------------

try:
    pv_df = pd.read_csv(PV_CSV)
    wind_df = pd.read_csv(WIND_CSV)
except Exception as e:
    print("!!! Error loading CSVs:", repr(e))
    # let it crash loudly at startup – easier to debug
    raise

print("PV columns:", list(pv_df.columns))
print("WIND columns:", list(wind_df.columns))

# parse datetime columns
for df, time_col, label in [
    (pv_df, PV_TIME_COL, "PV"),
    (wind_df, WIND_TIME_COL, "WIND"),
]:
    if time_col not in df.columns:
        raise RuntimeError(f"{label} time column '{time_col}' not found in CSV")
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

# -------------------------------------------------
# 4) Pydantic model
# -------------------------------------------------

class ForecastPoint(BaseModel):
    time: datetime
    tech: Literal["pv", "wind"]
    actual: Optional[float]
    forecast: float

# -------------------------------------------------
# 5) Endpoint
# -------------------------------------------------

@app.get("/Backend/forecast", response_model=List[ForecastPoint])
def get_forecast(
    tech: Literal["pv", "wind"] = Query(...),
    start: datetime = Query(...),
    end: datetime = Query(...),
):
    # strip timezone info so we compare against naive pandas timestamps
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.replace(tzinfo=None)

    if start >= end:
        raise HTTPException(status_code=400, detail="start must be < end")

    try:
        if tech == "pv":
            df = pv_df
            time_col = PV_TIME_COL
            fc_col = PV_FORECAST_COL
            actual_col = PV_ACTUAL_COL
        else:
            df = wind_df
            time_col = WIND_TIME_COL
            fc_col = WIND_FORECAST_COL
            actual_col = WIND_ACTUAL_COL

        # sanity checks – if these fail you'll see clear errors in the uvicorn log
        for col_name in (time_col, fc_col):
            if col_name not in df.columns:
                raise RuntimeError(f"Column '{col_name}' not found in {tech} CSV")

        # slice time window
        mask = (df[time_col] >= start) & (df[time_col] <= end)
        sub = df.loc[mask].copy().sort_values(time_col)

        points: list[ForecastPoint] = []
        for _, row in sub.iterrows():
            # forecast (must exist, otherwise we'd have raised above)
            fc_val = float(row[fc_col])

            # actual is optional
            actual_val: Optional[float] = None
            if actual_col is not None and actual_col in sub.columns:
                v = row[actual_col]
                if pd.notna(v):
                    actual_val = float(v)

            points.append(
                ForecastPoint(
                    time=row[time_col],
                    tech=tech,
                    actual=actual_val,
                    forecast=fc_val,
                )
            )

        return points

    except Exception as e:
        print("Error in /Backend/forecast:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
