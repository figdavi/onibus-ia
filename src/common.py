"""Shared functions used by both notebooks (P1 temporal, P2 spatial).

Contract (Portuguese aliases at the bottom keep the names agreed with the pair):

- ``prepare(df) -> df``            remove outliers and low-quality segments
- ``split(df) -> train, val, test``(P1) temporal split by service date
- ``build_features(df, reference) -> X, y`` (P2) model inputs and target
- ``evaluate(model, X, y) -> dict``(P2) MAE, RMSE and R2
- ``MedianBaseline``               (P2) reference model
- ``scaled(model)``                (P2) imputer + scaler wrapper for Ridge and MLP

Typical use in a notebook::

    df = prepare(pd.read_parquet(PATH))
    train, val, test = split(df)
    X_train, y_train = build_features(train, reference=train)
    X_test, y_test = build_features(test, reference=train)
    model = scaled(Ridge()).fit(X_train, y_train)
    evaluate(model, X_test, y_test)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "travel_time_s"

CARNIVAL_START = pd.Timestamp("2019-03-02")
CARNIVAL_END = pd.Timestamp("2019-03-06")

DIRECTIONS = ["Leblon", "Usina"]

MIN_DAY_FRACTION = 0.25
MIN_SPEED_KMH = 3.0
MAX_SPEED_KMH = 60.0
MIN_TRAVEL_TIME_S = 10.0

NUMERIC_FEATURES = [
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_carnival",
    "stop_seq",
    "distance_m",
    "lag_previous_segment_s",
    "lag_previous_bus_s",
    "minutes_since_previous_bus",
    "recent_buses_mean_s",
    "segment_median_s",
    "trip_pace_last",
    "trip_pace_mean",
]

RECENT_BUSES = 3


class Regressor(Protocol):
    """Anything with a scikit-learn style ``predict``."""

    def predict(self, X: pd.DataFrame) -> Any: ...


# --------------------------------------------------------------------------- #
# P1: preparation and split
# --------------------------------------------------------------------------- #


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and low-quality data with fixed physical rules.

    Rules use fixed thresholds instead of percentiles of the whole table, so no
    information from the test period leaks into cleaning. Numbers below were
    measured on line 415 (2019-01-25 to 2019-03-21):

    1. Sparse days: days with fewer than 25% of the median segments per day
       (only 2019-03-10, a GPS outage) are dropped.
    2. Implausible speed: segment speed outside 3-60 km/h (0.55% of rows).
       Below 3 km/h the bus is parked (layover); above 60 km/h is a GPS error.
    3. Too short: travel time under 10 s (0.06%), a stop-and-go artifact.

    Pings more than 300 s apart are already rejected by the pipeline
    (``max_gap_s`` <= 300), so no extra gap rule is needed here.

    Args:
        df: Segments table from ``segments_415_v0.parquet``.

    Returns:
        Cleaned copy of ``df``. ``result.attrs["prepare_report"]`` holds the
        number of rows removed by each rule (for the report).
    """
    out = df.copy()
    report: dict[str, int] = {"rows_in": len(out)}

    per_day = out.groupby("service_date")[TARGET].transform("size")
    sparse_day = (
        per_day < MIN_DAY_FRACTION * out.groupby("service_date").size().median()
    )
    report["sparse_days"] = int(sparse_day.sum())
    out = out[~sparse_day]

    speed_kmh = out["distance_m"] / out[TARGET] * 3.6
    implausible = ~speed_kmh.between(MIN_SPEED_KMH, MAX_SPEED_KMH)
    report["implausible_speed"] = int(implausible.sum())
    out = out[~implausible]

    too_short = out[TARGET] < MIN_TRAVEL_TIME_S
    report["too_short"] = int(too_short.sum())
    out = out[~too_short].copy()

    report["rows_out"] = len(out)
    out.attrs["prepare_report"] = report
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split by ``service_date`` (owner: P1).

    Expected behavior:
        - train: 2019-01-25 to 2019-03-03;
        - validation: 2019-03-04 to 2019-03-07 (includes Carnival);
        - test: 2019-03-11 to 2019-03-20.

    Args:
        df: Prepared segments table.

    Returns:
        Train, validation and test tables.

    Raises:
        NotImplementedError: Until P1 implements it.
    """
    raise NotImplementedError("split() is owned by P1")


# --------------------------------------------------------------------------- #
# P2: features
# --------------------------------------------------------------------------- #


def previous_buses(df: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    """Departure and travel time of the k-th previous bus on the same segment.

    Args:
        df: Segments table with ``direction``, ``stop_seq`` and ``departure_ts``.
        k: 1 for the bus right before, 2 for the one before it, and so on.

    Returns:
        Departure timestamps and travel times (seconds), aligned with ``df``.
    """
    ordered = df.sort_values("departure_ts")
    by_segment = ordered.groupby(["direction", "stop_seq"])
    departure = by_segment["departure_ts"].shift(k).reindex(df.index)
    travel = by_segment[TARGET].shift(k).reindex(df.index)
    return departure, travel


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features that are known at departure time (no leakage).

    - ``lag_previous_segment_s``: travel time of the previous segment in the
      same trip (it ends exactly when the current one starts).
    - ``lag_previous_bus_s``: travel time of the same segment by the previous
      bus, only if that bus had already arrived at the current departure time.
    - ``minutes_since_previous_bus``: time since the previous bus left the same
      stop (headway).
    - ``recent_buses_mean_s``: mean travel time of the last ``RECENT_BUSES``
      buses on the same segment that had already arrived.

    Args:
        df: Segments table with ``departure_ts`` as datetime.

    Returns:
        Copy of ``df`` with the lag columns (NaN when unknown).
    """
    out = df.sort_values("departure_ts").copy()

    by_trip = out.sort_values(["trip_id", "stop_seq"]).groupby("trip_id")
    previous_seq = by_trip["stop_seq"].shift(1)
    previous_time = by_trip[TARGET].shift(1)
    consecutive = previous_seq == out.loc[previous_seq.index, "stop_seq"] - 1
    out["lag_previous_segment_s"] = previous_time.where(consecutive)

    known_travel = []
    for k in range(1, RECENT_BUSES + 1):
        departure, travel = previous_buses(out, k)
        arrival = departure + pd.to_timedelta(travel, unit="s")
        known_travel.append(travel.where(arrival <= out["departure_ts"]))
        if k == 1:
            headway = (out["departure_ts"] - departure).dt.total_seconds() / 60
            out["minutes_since_previous_bus"] = headway
    out["lag_previous_bus_s"] = known_travel[0]
    out["recent_buses_mean_s"] = pd.concat(known_travel, axis=1).mean(axis=1)
    return out


def segment_medians(reference: pd.DataFrame) -> pd.Series:
    """Median travel time per (direction, stop_seq) in the reference table.

    Args:
        reference: Training segments (never validation or test).

    Returns:
        Medians indexed by (direction, stop_seq).
    """
    medians = reference.groupby(["direction", "stop_seq"])[TARGET].median()
    return medians.rename("segment_median_s")


def add_profile_features(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add the typical time of each segment and how fast the trip is running.

    - ``segment_median_s``: median of the segment in ``reference`` (training).
    - ``trip_pace_last``: previous segment of the trip divided by its median
      (1.2 means the bus just ran 20% slower than usual).
    - ``trip_pace_mean``: mean of that ratio over all previous segments of the
      trip (is this trip slow today?).

    Args:
        df: Segments table.
        reference: Training segments used for the medians. Passing the test
            table here leaks the answer into the features.

    Returns:
        Copy of ``df`` with the three columns (NaN when unknown).
    """
    out = df.join(segment_medians(reference), on=["direction", "stop_seq"])
    ordered = out.sort_values(["trip_id", "stop_seq"])
    ratio = ordered[TARGET] / ordered["segment_median_s"]
    by_trip = ratio.groupby(ordered["trip_id"])
    consecutive = (
        ordered.groupby("trip_id")["stop_seq"].shift(1) == ordered["stop_seq"] - 1
    )
    out["trip_pace_last"] = by_trip.shift(1).where(consecutive)

    filled = ratio.fillna(0.0)
    valid = ratio.notna().astype(float)
    cumulative = filled.groupby(ordered["trip_id"]).cumsum() - filled
    count = valid.groupby(ordered["trip_id"]).cumsum() - valid
    out["trip_pace_mean"] = (cumulative / count.where(count > 0)).reindex(out.index)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour (raw and cyclic), weekday, weekend and Carnival flags.

    Args:
        df: Segments table with ``departure_ts`` as datetime.

    Returns:
        Copy of ``df`` with calendar columns.
    """
    out = df.copy()
    timestamp = pd.to_datetime(out["departure_ts"])
    hour_float = timestamp.dt.hour + timestamp.dt.minute / 60
    out["hour"] = timestamp.dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)
    out["day_of_week"] = timestamp.dt.day_of_week
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    day = timestamp.dt.normalize()
    out["is_carnival"] = day.between(CARNIVAL_START, CARNIVAL_END).astype(int)
    return out


def enrich(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """All feature columns added to the segments table (used by ETA too).

    Args:
        df: Segments table.
        reference: Training segments (see ``add_profile_features``).

    Returns:
        Copy of ``df`` with lag, profile and calendar columns.
    """
    with_lags = add_lag_features(df)
    return add_calendar_features(add_profile_features(with_lags, reference))


def to_matrix(enriched: pd.DataFrame) -> pd.DataFrame:
    """Select the model columns and one-hot encode the direction.

    Args:
        enriched: Output of ``enrich`` (or a copy with modified columns).

    Returns:
        Feature matrix with fixed columns, as float.
    """
    direction = pd.Categorical(enriched["direction"], categories=DIRECTIONS)
    direction_dummies = pd.get_dummies(direction, prefix="direction", dtype=int)
    direction_dummies.index = enriched.index
    X = pd.concat([enriched[NUMERIC_FEATURES], direction_dummies], axis=1)
    return X.astype(float)


def build_features(
    df: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs ``X`` and target ``y`` (owner: P2).

    Categorical columns are one-hot encoded with fixed categories, so train and
    test always have the same columns. Lags may be NaN: tree models handle it
    (LightGBM natively, Random Forest after imputation) and ``scaled()`` imputes
    them for Ridge and MLP.

    Always pass the **training** table as ``reference``, also for validation
    and test: ``build_features(test, reference=train)``.

    Args:
        df: Segments table (any subset: train, validation or test).
        reference: Training segments, source of the segment medians.

    Returns:
        ``X`` (features) and ``y`` (``travel_time_s`` in seconds).
    """
    enriched = enrich(df, reference)
    return to_matrix(enriched), enriched[TARGET].astype(float)


def scaled(model: Any) -> Pipeline:
    """Wrap a model with median imputation and standardization.

    Use for Ridge and MLP (sensitive to scale and NaN). Random Forest only needs
    the imputer; LightGBM needs neither.

    Args:
        model: Unfitted scikit-learn style estimator.

    Returns:
        Pipeline imputer -> scaler -> model.
    """
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


# --------------------------------------------------------------------------- #
# P2: baseline and evaluation
# --------------------------------------------------------------------------- #


class MedianBaseline:
    """Median travel time per (direction, stop, hour, weekend) seen in training.

    Falls back to the median per (direction, stop) and then to the global median
    when a combination never appeared in training.
    """

    KEYS = ["direction_Usina", "stop_seq", "hour", "is_weekend"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MedianBaseline:
        """Store the training medians.

        Args:
            X: Features from ``build_features``.
            y: Target.

        Returns:
            The fitted baseline.
        """
        data = X[self.KEYS].assign(target=y.to_numpy())
        self.full_ = data.groupby(self.KEYS)["target"].median()
        self.segment_ = data.groupby(self.KEYS[:2])["target"].median()
        self.global_ = float(data["target"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the most specific median available.

        Args:
            X: Features from ``build_features``.

        Returns:
            Predicted travel times in seconds.
        """
        full_index = pd.MultiIndex.from_frame(X[self.KEYS])
        segment_index = pd.MultiIndex.from_frame(X[self.KEYS[:2]])
        prediction = self.full_.reindex(full_index).to_numpy()
        fallback = self.segment_.reindex(segment_index).to_numpy()
        prediction = np.where(np.isnan(prediction), fallback, prediction)
        return np.where(np.isnan(prediction), self.global_, prediction)


def evaluate(model: Regressor, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Regression metrics on a held-out set.

    Args:
        model: Fitted model with ``predict``.
        X: Features.
        y: True travel times in seconds.

    Returns:
        MAE and RMSE (seconds), R2 and number of rows.
    """
    predicted = np.asarray(model.predict(X), dtype=float)
    return {
        "MAE_s": round(float(mean_absolute_error(y, predicted)), 2),
        "RMSE_s": round(float(np.sqrt(mean_squared_error(y, predicted))), 2),
        "R2": round(float(r2_score(y, predicted)), 4),
        "n": int(len(y)),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{model_name: evaluate(...)}`` into a table sorted by MAE.

    Args:
        results: Metrics per model name.

    Returns:
        One row per model, best (lowest MAE) first.
    """
    return pd.DataFrame(results).T.sort_values("MAE_s")


# Portuguese aliases (names agreed with the pair).
preparar = prepare
features = build_features
avaliar = evaluate
