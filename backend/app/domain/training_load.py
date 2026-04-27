"""Training load metrics: TRIMP, CTL, ATL, TSB, ACWR.

These metrics are the foundation of every modern endurance-training app
(Strava Fitness/Freshness, Garmin Training Load, TrainingPeaks PMC, WKO5).

The model used here is the standard "Performance Manager Chart":

    TRIMP_t  := per-session training impulse, computed from heart rate.
    CTL_t    := exponentially weighted moving average of TRIMP, alpha = 2/43.
    ATL_t    := exponentially weighted moving average of TRIMP, alpha = 2/8.
    TSB_t    := CTL_{t-1} - ATL_{t-1}  (form / freshness, pre-session).
    ACWR_t   := sum(load_{t-6..t}) / (sum(load_{t-27..t}) / 4)  (Gabbett).

By convention, CTL and ATL start at 0 on the first day of the series.
TSB on day 0 is undefined (no previous CTL/ATL); we report it as 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

# Karvonen TRIMP coefficients (Banister 1991, Edwards 1993). ------------------

#: Linear weight on the gender-neutral TRIMP exponential.
KARVONEN_TRIMP_LINEAR_COEFF: float = 0.64

#: Exponential factor in the Karvonen TRIMP formula.
KARVONEN_TRIMP_EXP_COEFF: float = 1.92

# EWMA decay constants. -------------------------------------------------------

#: CTL: ~42-day fitness window.
CTL_ALPHA: float = 2.0 / 43.0

#: ATL: ~7-day fatigue window.
ATL_ALPHA: float = 2.0 / 8.0

# ACWR window sizes (in days). ------------------------------------------------

#: Acute window: most recent 7 days (today included).
ACWR_ACUTE_WINDOW: int = 7

#: Chronic window: most recent 28 days (today included).
ACWR_CHRONIC_WINDOW: int = 28


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class TrainingLoadPoint(BaseModel):
    """The training-load state on a single day."""

    ctl: float = Field(..., description="Chronic training load (fitness).")
    atl: float = Field(..., description="Acute training load (fatigue).")
    tsb: float = Field(..., description="Training stress balance (form).")


class TrainingLoadSeries(BaseModel):
    """Training-load metrics computed over a series of consecutive days."""

    points: list[TrainingLoadPoint]


# -----------------------------------------------------------------------------
# Per-session TRIMP
# -----------------------------------------------------------------------------


def calculate_trimp_karvonen(
    duration_min: float,
    hr_avg: float,
    hr_max: int,
    hr_rest: int,
) -> float:
    """Return the Karvonen TRIMP score for one training session.

    Formula:
        ratio = (HR_avg - HR_rest) / (HR_max - HR_rest)        # clamped to [0, 1]
        TRIMP = duration_min * ratio * 0.64 * exp(1.92 * ratio)

    Args:
        duration_min: Session duration, in minutes. Must be >= 0.
        hr_avg: Average heart rate during the session, in bpm.
        hr_max: Maximum heart rate, in bpm. Used to compute heart-rate reserve.
        hr_rest: Resting heart rate, in bpm.

    Returns:
        The TRIMP score (0 if HR_avg <= HR_rest, since the session was sub-aerobic).

    Raises:
        ValueError: If hr_max <= hr_rest, or if duration is negative.
    """
    if duration_min < 0:
        raise ValueError(f"duration_min must be >= 0 (got {duration_min}).")
    if hr_max <= hr_rest:
        raise ValueError(f"hr_max ({hr_max}) must be strictly greater than hr_rest ({hr_rest}).")

    if hr_avg <= hr_rest:
        # Sub-aerobic session — by convention, no TRIMP credit.
        return 0.0

    reserve = hr_max - hr_rest
    ratio = (hr_avg - hr_rest) / reserve
    ratio = max(0.0, min(ratio, 1.0))

    return (
        duration_min
        * ratio
        * KARVONEN_TRIMP_LINEAR_COEFF
        * math.exp(KARVONEN_TRIMP_EXP_COEFF * ratio)
    )


# -----------------------------------------------------------------------------
# CTL / ATL / TSB over a series of days
# -----------------------------------------------------------------------------


def _ewma(values: Sequence[float], alpha: float) -> list[float]:
    """Compute the exponentially weighted moving average, starting from 0.

    EWMA recurrence:
        s_0 = 0
        s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    """
    smoothed: list[float] = []
    previous = 0.0
    for x in values:
        current = alpha * x + (1.0 - alpha) * previous
        smoothed.append(current)
        previous = current
    return smoothed


def compute_training_load_series(
    daily_trimp: Sequence[float],
) -> TrainingLoadSeries:
    """Return CTL, ATL and TSB for every day of a TRIMP series.

    The series is assumed to be one entry per day, ordered chronologically.
    Rest days should be passed as 0 (do not skip them).

    Args:
        daily_trimp: Per-day TRIMP scores.

    Returns:
        A TrainingLoadSeries with one TrainingLoadPoint per input day.
    """
    ctl = _ewma(daily_trimp, alpha=CTL_ALPHA)
    atl = _ewma(daily_trimp, alpha=ATL_ALPHA)

    points: list[TrainingLoadPoint] = []
    for i in range(len(daily_trimp)):
        # TSB uses the previous day's values (form before today's session).
        tsb = (ctl[i - 1] - atl[i - 1]) if i > 0 else 0.0
        points.append(TrainingLoadPoint(ctl=ctl[i], atl=atl[i], tsb=tsb))

    return TrainingLoadSeries(points=points)


# -----------------------------------------------------------------------------
# ACWR (Gabbett injury-risk model)
# -----------------------------------------------------------------------------


def compute_acwr(daily_load: Sequence[float]) -> list[float]:
    """Return the Acute:Chronic Workload Ratio for every day of a load series.

    Formula (Gabbett 2016, weekly-normalized variant):
        ACWR_t = mean(load over acute window) / mean(load over chronic window)

    On early days where the full windows are not yet available, we use the
    available history (matches pandas `.rolling(window, min_periods=1)`).

    Returns 0.0 on days where the chronic window has zero load
    (no division-by-zero noise).
    """
    n = len(daily_load)
    result: list[float] = []
    for i in range(n):
        acute_start = max(0, i - ACWR_ACUTE_WINDOW + 1)
        chronic_start = max(0, i - ACWR_CHRONIC_WINDOW + 1)

        acute_window = daily_load[acute_start : i + 1]
        chronic_window = daily_load[chronic_start : i + 1]

        acute_avg = sum(acute_window) / len(acute_window)
        chronic_avg = sum(chronic_window) / len(chronic_window)

        if chronic_avg == 0:
            result.append(0.0)
        else:
            result.append(acute_avg / chronic_avg)
    return result
