"""
FastAPI endpoints — GET and POST for all three ML models.

Routes:
    GET/POST /predict/linear    — Linear Regression
    GET/POST /predict/forest    — Random Forest
    GET/POST /predict/deep      — Deep Learning (MLP)
    GET      /health            — Health check
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Literal, Annotated

import src.models.linear_regression as lr_model
import src.models.random_forest as rf_model
import src.models.deep_learning as dl_model

app = FastAPI(
    title="Tips Prediction API",
    description="REST API exposing three ML models to predict restaurant tip amounts.",
    version="2.0.0",
)

# ── Input / Output schemas ─────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    total_bill: float = Field(..., gt=0, description="Total bill amount in $",
                              json_schema_extra={"example": 18.5})
    size: int = Field(..., ge=1, le=6, description="Party size (1–6)",
                      json_schema_extra={"example": 3})
    sex: Literal["Male", "Female"]
    smoker: Literal["Yes", "No"]
    day: Literal["Fri", "Sat", "Sun", "Thur"]
    time: Literal["Dinner", "Lunch"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_bill": 18.5, "size": 3,
                "sex": "Male", "smoker": "No",
                "day": "Sat", "time": "Dinner",
            }
        }
    }


class PredictionOutput(BaseModel):
    model: str
    predicted_tip: float
    currency: str = "$"


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Linear Regression ──────────────────────────────────────────────────────────

@app.get("/predict/linear", response_model=PredictionOutput, tags=["Linear Regression"])
def predict_linear_get(
    total_bill: Annotated[float, Query(gt=0)] = 18.5,
    size: Annotated[int, Query(ge=1, le=6)] = 3,
    sex: Literal["Male", "Female"] = "Male",
    smoker: Literal["Yes", "No"] = "No",
    day: Literal["Fri", "Sat", "Sun", "Thur"] = "Sat",
    time: Literal["Dinner", "Lunch"] = "Dinner",
):
    """Predict tip with Linear Regression via query parameters."""
    tip = lr_model.predict(total_bill, size, sex, smoker, day, time)
    return PredictionOutput(model="linear_regression", predicted_tip=tip)


@app.post("/predict/linear", response_model=PredictionOutput, tags=["Linear Regression"])
def predict_linear_post(data: PredictionInput):
    """Predict tip with Linear Regression via JSON body."""
    tip = lr_model.predict(data.total_bill, data.size, data.sex, data.smoker, data.day, data.time)
    return PredictionOutput(model="linear_regression", predicted_tip=tip)


# ── Random Forest ──────────────────────────────────────────────────────────────

@app.get("/predict/forest", response_model=PredictionOutput, tags=["Random Forest"])
def predict_forest_get(
    total_bill: Annotated[float, Query(gt=0)] = 18.5,
    size: Annotated[int, Query(ge=1, le=6)] = 3,
    sex: Literal["Male", "Female"] = "Male",
    smoker: Literal["Yes", "No"] = "No",
    day: Literal["Fri", "Sat", "Sun", "Thur"] = "Sat",
    time: Literal["Dinner", "Lunch"] = "Dinner",
):
    """Predict tip with Random Forest via query parameters."""
    tip = rf_model.predict(total_bill, size, sex, smoker, day, time)
    return PredictionOutput(model="random_forest", predicted_tip=tip)


@app.post("/predict/forest", response_model=PredictionOutput, tags=["Random Forest"])
def predict_forest_post(data: PredictionInput):
    """Predict tip with Random Forest via JSON body."""
    tip = rf_model.predict(data.total_bill, data.size, data.sex, data.smoker, data.day, data.time)
    return PredictionOutput(model="random_forest", predicted_tip=tip)


# ── Deep Learning ──────────────────────────────────────────────────────────────

@app.get("/predict/deep", response_model=PredictionOutput, tags=["Deep Learning"])
def predict_deep_get(
    total_bill: Annotated[float, Query(gt=0)] = 18.5,
    size: Annotated[int, Query(ge=1, le=6)] = 3,
    sex: Literal["Male", "Female"] = "Male",
    smoker: Literal["Yes", "No"] = "No",
    day: Literal["Fri", "Sat", "Sun", "Thur"] = "Sat",
    time: Literal["Dinner", "Lunch"] = "Dinner",
):
    """Predict tip with Deep Learning (MLP) via query parameters."""
    tip = dl_model.predict(total_bill, size, sex, smoker, day, time)
    return PredictionOutput(model="deep_learning", predicted_tip=tip)


@app.post("/predict/deep", response_model=PredictionOutput, tags=["Deep Learning"])
def predict_deep_post(data: PredictionInput):
    """Predict tip with Deep Learning (MLP) via JSON body."""
    tip = dl_model.predict(data.total_bill, data.size, data.sex, data.smoker, data.day, data.time)
    return PredictionOutput(model="deep_learning", predicted_tip=tip)
