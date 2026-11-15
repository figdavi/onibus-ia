"""Shared preprocessing of the stop-to-stop segments table.

Steps, in the order they must be applied:

1. ``add_calendar``: hour, weekday, day type and the Carnival flag.
2. ``filter_quality``: drop segments timed from pings that were too far apart.
3. ``temporal_split``: train on the past, validate on Carnival week, test on
   the future (no shuffling).
4. ``fit_outlier_bounds`` on the training set only, then ``apply_outlier_bounds``
   to every split. Segments with too few training samples are dropped, because
   neither the bounds nor the models can be estimated for them.

``prepare`` runs all of it and returns the three splits.

Usage:
    from prep import prepare
    splits = prepare(pd.read_parquet("segments_415_v0.parquet"))
    splits.train, splits.val, splits.test
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

SEGMENT_KEY = ["line", "direction", "stop_seq"]

CARNIVAL_START = date(2019, 3, 2)
CARNIVAL_END = date(2019, 3, 6)

TRAIN_END = date(2019, 3, 3)
VAL_START, VAL_END = date(2019, 3, 4), date(2019, 3, 7)
TEST_START, TEST_END = date(2019, 3, 11), date(2019, 3, 20)

MAX_GAP_S = 240.0
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99
MIN_TRAIN_SAMPLES = 30

WEEKDAY_NAMES = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


@dataclass(frozen=True)
class Splits:
    """Train, validation and test sets, already cleaned."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def add_calendar(segments: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day and calendar columns derived from ``departure_ts``.

    Args:
        segments: Segments table with ``departure_ts`` and ``service_date``.

    Returns:
        Copy with ``hour``, ``weekday`` (0 = Monday), ``is_weekend``,
        ``is_carnival`` and ``day_type`` (útil / sábado / domingo / carnaval).
    """
    out = segments.copy()
    ts = pd.to_datetime(out["departure_ts"])
    service_date = pd.to_datetime(out["service_date"]).dt.date
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.day_of_week
    out["is_weekend"] = out["weekday"] >= 5
    out["is_carnival"] = (service_date >= CARNIVAL_START) & (
        service_date <= CARNIVAL_END
    )
    day_type = pd.Series("útil", index=out.index)
    day_type[out["weekday"] == 5] = "sábado"
    day_type[out["weekday"] == 6] = "domingo"
    day_type[out["is_carnival"]] = "carnaval"
    out["day_type"] = day_type
    return out


def filter_quality(
    segments: pd.DataFrame, max_gap_s: float = MAX_GAP_S
) -> pd.DataFrame:
    """Keep segments whose worst bracketing ping gap is at most ``max_gap_s``.

    Args:
        segments: Segments table with ``max_gap_s``.
        max_gap_s: Largest accepted gap between the pings used to time a stop.

    Returns:
        Filtered copy.
    """
    return segments[segments["max_gap_s"] <= max_gap_s].copy()


def temporal_split(
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by service date: train <= 03/03, val 04-07/03, test 11-20/03.

    08-10/03 are left out as a buffer (10/03 is almost empty) and 21/03 is a
    partial day.

    Args:
        segments: Segments table with ``service_date``.

    Returns:
        Train, validation and test frames.
    """
    day = pd.to_datetime(segments["service_date"]).dt.date
    train = segments[day <= TRAIN_END]
    val = segments[(day >= VAL_START) & (day <= VAL_END)]
    test = segments[(day >= TEST_START) & (day <= TEST_END)]
    return train.copy(), val.copy(), test.copy()


def fit_outlier_bounds(
    train: pd.DataFrame,
    lower: float = LOWER_QUANTILE,
    upper: float = UPPER_QUANTILE,
    min_samples: int = MIN_TRAIN_SAMPLES,
) -> pd.DataFrame:
    """Per-segment travel-time bounds estimated on the training set.

    Args:
        train: Training segments.
        lower: Lower quantile.
        upper: Upper quantile.
        min_samples: Segments with fewer training rows get no bounds.

    Returns:
        One row per segment key with ``tt_low``, ``tt_high`` and ``n_train``.
    """
    grouped = train.groupby(SEGMENT_KEY)["travel_time_s"]
    bounds = pd.DataFrame(
        {
            "tt_low": grouped.quantile(lower),
            "tt_high": grouped.quantile(upper),
            "n_train": grouped.size(),
        }
    )
    return bounds[bounds["n_train"] >= min_samples].reset_index()


def apply_outlier_bounds(segments: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Drop rows outside their segment bounds, or of segments without bounds.

    Args:
        segments: Any split.
        bounds: Output of ``fit_outlier_bounds``.

    Returns:
        Filtered copy (same columns as the input).
    """
    merged = segments.merge(bounds, on=SEGMENT_KEY, how="inner")
    inside = merged["travel_time_s"].between(merged["tt_low"], merged["tt_high"])
    return merged[inside].drop(columns=["tt_low", "tt_high", "n_train"])


def prepare(segments: pd.DataFrame, max_gap_s: float = MAX_GAP_S) -> Splits:
    """Full preprocessing: calendar, quality filter, split and outliers.

    Args:
        segments: Raw output of ``rio_segments.py``.
        max_gap_s: Quality threshold passed to ``filter_quality``.

    Returns:
        Cleaned train, validation and test sets.
    """
    clean = filter_quality(add_calendar(segments), max_gap_s)
    train, val, test = temporal_split(clean)
    bounds = fit_outlier_bounds(train)
    return Splits(
        train=apply_outlier_bounds(train, bounds).reset_index(drop=True),
        val=apply_outlier_bounds(val, bounds).reset_index(drop=True),
        test=apply_outlier_bounds(test, bounds).reset_index(drop=True),
    )
