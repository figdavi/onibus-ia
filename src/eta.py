"""ETA to stop X: chain segment predictions from an origin stop.

The segment models answer "how long from stop k to k+1?". A passenger asks
"the bus is at stop k now, when does it reach stop k+h?". The answer is the
sum of the next h segment predictions, made **with only what is known when the
bus is at stop k** (time t0):

- the first segment uses its real features (all known at t0);
- later segments use the previous *prediction* as ``lag_previous_segment_s``,
  the departure time ``t0 + predicted elapsed`` for the calendar columns, the
  trip pace measured up to t0, and only previous buses that arrived by t0.

The predicted arrival is compared with the real arrival from the GPS (test
days the model never saw).

Typical use in a notebook::

    models = {"Baseline": baseline, "Random Forest": rf}
    eta.evaluate_eta(models, test, reference=train)
    eta.replay(models, test, reference=train, trip_id=..., origin_seq=...)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import common

HORIZONS = (1, 3, 5, 10)
MAX_ORIGINS = 5000


def chains(enriched: pd.DataFrame, horizon: int) -> np.ndarray:
    """Rows of every origin followed by ``horizon - 1`` consecutive segments.

    Args:
        enriched: Output of ``common.enrich`` with a default RangeIndex.
        horizon: Number of segments in each chain.

    Returns:
        Integer array (n_origins, horizon) of row positions in ``enriched``.
    """
    keys = pd.MultiIndex.from_frame(enriched[["trip_id", "stop_seq"]])
    position = pd.Series(np.arange(len(enriched)), index=keys)
    if not position.index.is_unique:
        raise ValueError("(trip_id, stop_seq) must be unique")
    columns = []
    for k in range(horizon):
        shifted = pd.MultiIndex.from_arrays(
            [enriched["trip_id"], enriched["stop_seq"] + k]
        )
        columns.append(position.reindex(shifted).to_numpy())
    matrix = np.column_stack(columns)
    complete = ~np.isnan(matrix).any(axis=1)
    return matrix[complete].astype(int)


def _known_previous_buses(enriched: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Departure and travel of the last buses on each row's segment."""
    return [
        common.previous_buses(enriched, k) for k in range(1, common.RECENT_BUSES + 1)
    ]


def predict_chain(
    model: common.Regressor,
    enriched: pd.DataFrame,
    chain: np.ndarray,
    previous: list[tuple[pd.Series, pd.Series]],
) -> np.ndarray:
    """Cumulative predicted seconds from the origin along each chain.

    Args:
        model: Fitted segment model (any ``predict`` on ``to_matrix`` columns).
        enriched: Output of ``common.enrich`` with a default RangeIndex.
        chain: Row positions, shape (n_origins, horizon); column 0 is the origin.
        previous: Output of ``_known_previous_buses(enriched)``.

    Returns:
        Array (n_origins, horizon): predicted time from the origin departure
        to the end of segment ``k``.
    """
    origin = enriched.iloc[chain[:, 0]]
    t0 = origin["departure_ts"].to_numpy()
    pace_last = origin["trip_pace_last"].to_numpy()
    pace_mean = origin["trip_pace_mean"].to_numpy()

    elapsed = np.zeros(len(chain))
    last_prediction = np.full(len(chain), np.nan)
    cumulative = np.zeros(chain.shape, dtype=float)
    for k in range(chain.shape[1]):
        rows = enriched.iloc[chain[:, k]].copy()
        if k > 0:
            departure = t0 + pd.to_timedelta(elapsed, unit="s").to_numpy()
            rows["departure_ts"] = departure
            rows = common.add_calendar_features(rows)
            rows["lag_previous_segment_s"] = last_prediction
            rows["trip_pace_last"] = pace_last
            rows["trip_pace_mean"] = pace_mean

            known_travel = []
            for j, (bus_departure, bus_travel) in enumerate(previous):
                dep = bus_departure.iloc[chain[:, k]].to_numpy()
                travel = bus_travel.iloc[chain[:, k]].to_numpy()
                arrival = dep + pd.to_timedelta(travel, unit="s").to_numpy()
                known = arrival <= t0
                known_travel.append(np.where(known, travel, np.nan))
                if j == 0:
                    headway = (departure - dep) / np.timedelta64(1, "m")
                    rows["minutes_since_previous_bus"] = np.where(
                        dep <= t0, headway, np.nan
                    )
            stacked = np.column_stack(known_travel)
            rows["lag_previous_bus_s"] = stacked[:, 0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
                rows["recent_buses_mean_s"] = np.nanmean(stacked, axis=1)

        prediction = np.asarray(model.predict(common.to_matrix(rows)), dtype=float)
        elapsed = elapsed + prediction
        cumulative[:, k] = elapsed
        last_prediction = prediction
    return cumulative


def evaluate_eta(
    models: dict[str, common.Regressor],
    test: pd.DataFrame,
    reference: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    max_origins: int = MAX_ORIGINS,
    seed: int = 42,
) -> pd.DataFrame:
    """ETA error by number of stops ahead, for each model.

    All models and horizons use the same origins (those with at least
    ``max(horizons)`` consecutive valid segments), so the rows are comparable.

    Args:
        models: Fitted segment models by display name.
        test: Prepared test segments.
        reference: Training segments (for the segment medians).
        horizons: Stops ahead to evaluate.
        max_origins: Random sample size, to keep it fast.
        seed: Sampling seed.

    Returns:
        One row per (model, stops ahead): MAE in seconds and minutes, the mean
        real time to that stop and the number of origins.
    """
    enriched = common.enrich(test, reference).reset_index(drop=True)
    horizon = max(horizons)
    all_chains = chains(enriched, horizon)
    rng = np.random.default_rng(seed)
    if len(all_chains) > max_origins:
        all_chains = all_chains[rng.choice(len(all_chains), max_origins, replace=False)]

    travel = enriched[common.TARGET].to_numpy()[all_chains]
    actual = np.cumsum(travel, axis=1)
    previous = _known_previous_buses(enriched)

    rows = []
    for name, model in models.items():
        predicted = predict_chain(model, enriched, all_chains, previous)
        for h in horizons:
            error = np.abs(predicted[:, h - 1] - actual[:, h - 1])
            rows.append(
                {
                    "model": name,
                    "stops_ahead": h,
                    "MAE_s": round(float(error.mean()), 1),
                    "MAE_min": round(float(error.mean()) / 60, 2),
                    "mean_real_min": round(float(actual[:, h - 1].mean()) / 60, 1),
                    "n_origins": len(all_chains),
                }
            )
    return pd.DataFrame(rows)


def replay(
    models: dict[str, common.Regressor],
    test: pd.DataFrame,
    reference: pd.DataFrame,
    trip_id: str,
    origin_seq: int,
    horizon: int,
) -> pd.DataFrame:
    """Predicted vs. real arrival at each next stop for one real trip.

    Args:
        models: Fitted segment models by display name.
        test: Prepared test segments.
        reference: Training segments.
        trip_id: Trip to replay.
        origin_seq: ``stop_seq`` of the origin segment.
        horizon: Number of segments ahead (must be valid in this trip).

    Returns:
        One row per stop reached: stop name, real arrival and one predicted
        arrival column per model.

    Raises:
        ValueError: If the trip has no ``horizon`` consecutive segments there.
    """
    enriched = common.enrich(test, reference).reset_index(drop=True)
    all_chains = chains(enriched, horizon)
    origin_rows = enriched.iloc[all_chains[:, 0]]
    match = (origin_rows["trip_id"] == trip_id).to_numpy() & (
        origin_rows["stop_seq"] == origin_seq
    ).to_numpy()
    if not match.any():
        raise ValueError(f"no {horizon} consecutive segments from {origin_seq}")
    chain = all_chains[match]
    previous = _known_previous_buses(enriched)

    segments = enriched.iloc[chain[0]]
    t0 = segments["departure_ts"].iloc[0]
    real = np.cumsum(segments[common.TARGET].to_numpy())
    result = pd.DataFrame(
        {
            "stop_seq": segments["stop_seq"].to_numpy() + 1,
            "stop": segments["stop_to"].to_numpy(),
            "real": t0 + pd.to_timedelta(real, unit="s"),
        }
    )
    for name, model in models.items():
        predicted = predict_chain(model, enriched, chain, previous)[0]
        result[name] = t0 + pd.to_timedelta(predicted, unit="s")
    return result
