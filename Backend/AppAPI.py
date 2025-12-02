# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI()

class ForecastPoint(BaseModel):
    time: datetime
    pv_MW: float
    wind_MW: float

@app.get("/api/forecast", response_model=List[ForecastPoint])
def get_forecast():
    # here you call your trained PV/wind models
    # and return time series
    ...
