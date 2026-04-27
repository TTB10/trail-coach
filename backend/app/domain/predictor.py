"""Race finish-time prediction (Riegel V4.1 model).

This module predicts how long a runner will take to complete a target race,
based on a known personal record and the runner's current physiological state.

The model is a six-layer adjusted Riegel formula:

    T_target = T_record_corrected * (effort_target / effort_record)^k

Where:
    - effort = distance + D+ / 100  (see app.domain.effort)
    - T_record_corrected = T_record / speed_ratio  (VMA degradation)
    - k starts at a baseline depending on experience and is adjusted by:
        * fitness deficit  (low CTL vs. what the race demands)
        * extreme-effort penalty  (above 100 / 200 effort-km)
        * gap penalty  (target much harder than personal record)
        * strength training bonus  (reduces k slightly)
    - k is clamped to [1.02, 1.45] for safety.

Each layer is a small, testable function.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Profile enums
# -----------------------------------------------------------------------------


class ExperienceLevel(StrEnum):
    """Athlete's overall ultra-trail experience level."""

    BEGINNER = "beginner"
    AMATEUR = "amateur"
    CONFIRMED = "confirmed"
    ELITE = "elite"


class StrengthTrainingFrequency(StrEnum):
    """How often the athlete does strength / cross-training sessions."""

    NEVER = "never"
    ONCE_PER_WEEK = "once_per_week"
    TWICE_OR_MORE = "twice_or_more"


# -----------------------------------------------------------------------------
# Riegel exponent — baseline by experience
# -----------------------------------------------------------------------------

#: Base value of the Riegel exponent k for each experience level.
BASELINE_EXPONENT_BY_EXPERIENCE: dict[ExperienceLevel, float] = {
    ExperienceLevel.ELITE: 1.04,
    ExperienceLevel.CONFIRMED: 1.06,
    ExperienceLevel.AMATEUR: 1.09,
    ExperienceLevel.BEGINNER: 1.12,
}

# Safety bounds for the Riegel exponent. --------------------------------------
MIN_EXPONENT: float = 1.02
MAX_EXPONENT: float = 1.45

# Layer 3: fitness deficit ----------------------------------------------------

#: How aggressively a fitness deficit pushes k upward (multiplier per ratio).
FITNESS_DEFICIT_PENALTY: float = 0.15

#: A target race needs roughly 35% of its effort as CTL to be "well-prepared".
CTL_DEMAND_RATIO: float = 0.35

# Layer 4: extreme-effort penalty ---------------------------------------------

EFFORT_THRESHOLD_HARD_KM: float = 100.0
EFFORT_PENALTY_HARD: float = 0.0003

EFFORT_THRESHOLD_EXTREME_KM: float = 200.0
EFFORT_PENALTY_EXTREME: float = 0.0005

# Layer 5: gap penalty (target much harder than record) -----------------------

GAP_THRESHOLD: float = 3.0  # When target effort > 3 * record effort.
GAP_PENALTY_PER_UNIT: float = 0.015

# Layer 6: strength-training bonus --------------------------------------------

STRENGTH_BONUS_BY_FREQUENCY: dict[StrengthTrainingFrequency, float] = {
    StrengthTrainingFrequency.NEVER: 0.0,
    StrengthTrainingFrequency.ONCE_PER_WEEK: -0.01,
    StrengthTrainingFrequency.TWICE_OR_MORE: -0.02,
}

# -----------------------------------------------------------------------------
# Speed (VMA) degradation
# -----------------------------------------------------------------------------

#: Curve sharpness for the speed-ratio: smaller = gentler degradation.
SPEED_RATIO_EXPONENT: float = 0.25

#: Floor of the speed ratio: even with very low CTL, we don't degrade below 85%.
SPEED_RATIO_FLOOR: float = 0.85

# -----------------------------------------------------------------------------
# Prediction range (optimistic / pessimistic)
# -----------------------------------------------------------------------------

#: Base error margin (4%) plus a term that grows with effort.
ERROR_BASE: float = 0.04
ERROR_PER_EFFORT_KM: float = 1.0 / 2000.0  # +0.05% per effort-km.
ERROR_MAX: float = 0.20

#: Experience scales the error: less experienced = more variability.
ERROR_MULTIPLIER_BY_EXPERIENCE: dict[ExperienceLevel, float] = {
    ExperienceLevel.ELITE: 1.0,
    ExperienceLevel.CONFIRMED: 1.0,
    ExperienceLevel.AMATEUR: 1.2,
    ExperienceLevel.BEGINNER: 1.5,
}

#: Optimistic side: the prediction can drop by error * 0.6.
OPTIMISTIC_FACTOR: float = 0.6

#: Pessimistic side: the prediction can extend by error * 1.3 (asymmetric).
PESSIMISTIC_FACTOR: float = 1.3

# -----------------------------------------------------------------------------
# Layer 1 — Speed (VMA) degradation
# -----------------------------------------------------------------------------


def calculate_speed_degradation_factor(
    current_ctl: float,
    record_ctl: float,
) -> float:
    """Return the multiplier to apply to the record speed.

    A speed ratio of 1.0 means the runner is in the same shape as at their
    record. A ratio of 0.85 (the floor) means severely detrained.

    Args:
        current_ctl: Current CTL (chronic training load) of the runner.
        record_ctl: CTL at the time of the personal record.

    Returns:
        The speed ratio in [SPEED_RATIO_FLOOR, 1.0].
    """
    if record_ctl <= 0:
        # No reference CTL → assume current shape is fine.
        return 1.0

    raw = min(current_ctl / record_ctl, 1.0)
    if raw <= 0:
        return SPEED_RATIO_FLOOR

    # Apply the curve and clamp to floor.
    ratio = float(raw**SPEED_RATIO_EXPONENT)
    return max(ratio, SPEED_RATIO_FLOOR)


# -----------------------------------------------------------------------------
# Layers 2-6 — Riegel exponent adjustments
# -----------------------------------------------------------------------------


def baseline_exponent_for_experience(experience: ExperienceLevel) -> float:
    """Return the starting Riegel exponent for the given experience level."""
    return BASELINE_EXPONENT_BY_EXPERIENCE[experience]


def fitness_deficit_adjustment(
    projected_ctl: float,
    target_effort: float,
) -> float:
    """Return the exponent bump caused by a fitness deficit.

    A target race demands a CTL of roughly target_effort * CTL_DEMAND_RATIO.
    If the projected CTL is below that, we add a penalty proportional to
    the relative deficit.

    Returns 0 if the runner is well-prepared.
    """
    required_ctl = target_effort * CTL_DEMAND_RATIO
    if required_ctl <= 0 or projected_ctl >= required_ctl:
        return 0.0

    deficit_ratio = (required_ctl - projected_ctl) / required_ctl
    return deficit_ratio * FITNESS_DEFICIT_PENALTY


def extreme_effort_adjustment(target_effort: float) -> float:
    """Return the exponent bump for very long efforts.

    Two thresholds: above 100 effort-km, a small per-km penalty kicks in.
    Above 200, an additional, steeper per-km penalty stacks on top.
    """
    bump = 0.0
    if target_effort > EFFORT_THRESHOLD_HARD_KM:
        bump += (target_effort - EFFORT_THRESHOLD_HARD_KM) * EFFORT_PENALTY_HARD
    if target_effort > EFFORT_THRESHOLD_EXTREME_KM:
        bump += (target_effort - EFFORT_THRESHOLD_EXTREME_KM) * EFFORT_PENALTY_EXTREME
    return bump


def gap_adjustment(target_effort: float, record_effort: float) -> float:
    """Return the exponent bump when the target is much harder than the record.

    If target_effort > 3 * record_effort, every unit beyond that ratio adds
    a small penalty (extrapolation gets risky).
    """
    if record_effort <= 0:
        return 0.0
    gap_ratio = target_effort / record_effort
    if gap_ratio <= GAP_THRESHOLD:
        return 0.0
    return (gap_ratio - GAP_THRESHOLD) * GAP_PENALTY_PER_UNIT


def strength_training_adjustment(frequency: StrengthTrainingFrequency) -> float:
    """Return the exponent bonus (negative number) for strength training."""
    return STRENGTH_BONUS_BY_FREQUENCY[frequency]


def adjust_riegel_exponent(
    experience: ExperienceLevel,
    projected_ctl: float,
    target_effort: float,
    record_effort: float,
    strength_training: StrengthTrainingFrequency,
) -> float:
    """Compute the final Riegel exponent k by stacking all adjustment layers.

    The result is clamped to [MIN_EXPONENT, MAX_EXPONENT] for safety.
    """
    k = baseline_exponent_for_experience(experience)
    k += fitness_deficit_adjustment(projected_ctl, target_effort)
    k += extreme_effort_adjustment(target_effort)
    k += gap_adjustment(target_effort, record_effort)
    k += strength_training_adjustment(strength_training)
    return max(MIN_EXPONENT, min(k, MAX_EXPONENT))


# -----------------------------------------------------------------------------
# Result schema
# -----------------------------------------------------------------------------


class RacePrediction(BaseModel):
    """The full result of a race finish-time prediction.

    The prediction is exposed alongside the inputs that drove the adjustments,
    so the user can understand why the time is what it is.
    """

    predicted_minutes: float = Field(..., description="Most likely finish time, in minutes.")
    optimistic_minutes: float = Field(..., description="Optimistic edge of the range.")
    pessimistic_minutes: float = Field(..., description="Pessimistic edge of the range.")
    error_margin: float = Field(..., description="Relative error used for the range, in [0, 1].")
    riegel_exponent: float = Field(..., description="Final Riegel exponent k after adjustments.")
    record_effort: float = Field(
        ..., description="Equivalent effort of the personal record, in km."
    )
    target_effort: float = Field(..., description="Equivalent effort of the target race, in km.")
    speed_ratio: float = Field(..., description="VMA degradation factor in [0.85, 1.0].")


# -----------------------------------------------------------------------------
# Prediction range
# -----------------------------------------------------------------------------


def calculate_error_margin(
    target_effort: float,
    experience: ExperienceLevel,
) -> float:
    """Return the relative error used to build the optimistic/pessimistic range.

    The error grows with effort (long races are less predictable) and is
    multiplied by an experience-dependent factor (less experience = more noise).
    Clamped to ERROR_MAX.
    """
    base = ERROR_BASE + target_effort * ERROR_PER_EFFORT_KM
    scaled = base * ERROR_MULTIPLIER_BY_EXPERIENCE[experience]
    return min(scaled, ERROR_MAX)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def predict_finish_time(
    record_time_min: float,
    record_effort: float,
    target_effort: float,
    experience: ExperienceLevel,
    strength_training: StrengthTrainingFrequency,
    current_ctl: float,
    record_ctl: float,
    projected_ctl: float,
) -> RacePrediction:
    """Predict the runner's finish time on a target race.

    The inputs combine three things:
        1. A reference performance: time and equivalent effort of a recent PR.
        2. The target race: equivalent effort.
        3. The runner's state: experience, strength training, current/projected CTL.

    Args:
        record_time_min: Personal record time, in minutes. Must be > 0.
        record_effort: Equivalent effort of the record course (see effort module).
        target_effort: Equivalent effort of the target course.
        experience: Athlete's experience level.
        strength_training: Athlete's strength-training frequency.
        current_ctl: Current chronic training load.
        record_ctl: Chronic training load at the time of the record.
        projected_ctl: Expected CTL on race day (after taper).

    Returns:
        A RacePrediction with the predicted time, the optimistic/pessimistic
        range, and the parameters that drove the model (for transparency).

    Raises:
        ValueError: If record_time_min, record_effort or target_effort is non-positive.
    """
    if record_time_min <= 0:
        raise ValueError(f"record_time_min must be > 0 (got {record_time_min}).")
    if record_effort <= 0:
        raise ValueError(f"record_effort must be > 0 (got {record_effort}).")
    if target_effort <= 0:
        raise ValueError(f"target_effort must be > 0 (got {target_effort}).")

    # Layer 1: speed degradation (VMA).
    speed_ratio = calculate_speed_degradation_factor(current_ctl, record_ctl)
    record_time_corrected = record_time_min / speed_ratio

    # Layers 2-6: stack the Riegel exponent adjustments.
    k = adjust_riegel_exponent(
        experience=experience,
        projected_ctl=projected_ctl,
        target_effort=target_effort,
        record_effort=record_effort,
        strength_training=strength_training,
    )

    # The Riegel formula itself.
    predicted = float(record_time_corrected * (target_effort / record_effort) ** k)

    # Range around the prediction.
    error = calculate_error_margin(target_effort, experience)
    optimistic = predicted * (1.0 - error * OPTIMISTIC_FACTOR)
    pessimistic = predicted * (1.0 + error * PESSIMISTIC_FACTOR)

    return RacePrediction(
        predicted_minutes=predicted,
        optimistic_minutes=optimistic,
        pessimistic_minutes=pessimistic,
        error_margin=error,
        riegel_exponent=k,
        record_effort=record_effort,
        target_effort=target_effort,
        speed_ratio=speed_ratio,
    )
