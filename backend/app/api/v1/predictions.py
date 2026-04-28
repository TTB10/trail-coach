"""Prediction endpoint — exposes the Riegel V4.1 model over HTTP."""

from __future__ import annotations

from fastapi import APIRouter

from app.domain.effort import calculate_corrected_effort
from app.domain.predictor import RacePrediction, predict_finish_time
from app.schemas.prediction import PredictionRequest

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/race", response_model=RacePrediction)
def predict_race(request: PredictionRequest) -> RacePrediction:
    """Predict the runner's finish time on a target race.

    The endpoint converts the physical parameters (distance + elevation) into
    the equivalent effort the model expects, then runs the Riegel V4.1
    prediction.
    """
    record_effort = calculate_corrected_effort(
        distance_km=request.record_distance_km,
        elevation_gain_m=request.record_elevation_gain_m,
    )
    target_effort = calculate_corrected_effort(
        distance_km=request.target_distance_km,
        elevation_gain_m=request.target_elevation_gain_m,
    )

    return predict_finish_time(
        record_time_min=request.record_time_min,
        record_effort=record_effort,
        target_effort=target_effort,
        experience=request.experience,
        strength_training=request.strength_training,
        current_ctl=request.current_ctl,
        record_ctl=request.record_ctl,
        projected_ctl=request.projected_ctl,
    )
