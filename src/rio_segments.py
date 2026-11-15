"""Build stop-to-stop bus travel-time segments from raw GPS pings (Rio de Janeiro).

Pipeline (one bus line at a time):

1. Route polyline per direction, from the stops file (file order == route order).
2. Project every ping onto each direction polyline: along-route position ``s``
   (meters) and lateral ``offset`` (meters).
3. Infer the direction of every ping with a Viterbi pass (penalizes large offsets,
   backward movement and direction switches) and split pings into trips.
4. Clean ``s`` inside each trip: drop physically impossible jumps, then apply a
   cumulative maximum so the position never goes backward.
5. Interpolate the passage time at every stop from the two pings that bracket it
   and build one row per segment (stop k -> stop k+1).
6. Validate the interpolation with a ping holdout (hide every other ping, predict
   its time from the remaining ones, measure the error).

Usage:
    uv run python src/rio_segments.py --line 415 --start 2019-02-11 --end 2019-02-17
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

DATA_DIR = Path(__file__).resolve().parents[1] / "dados"
GPS_PATH = DATA_DIR / "6-kaggle-rio" / "gps_rio_2019.parquet"
STOPS_PATH = DATA_DIR / "6-kaggle-rio" / "bus_stops_rio_de_janeiro.csv"
OUTPUT_DIR = DATA_DIR / "7-segments"

METERS_PER_DEG_LAT = 110_540.0
METERS_PER_DEG_LON_AT_EQUATOR = 111_320.0

MIN_STOPS_PER_DIRECTION = 5
MAX_OFFSET_M = 150.0
MAX_TRIP_GAP_S = 600.0
MAX_BRACKET_GAP_S = 300.0
STATIONARY_TOLERANCE_M = 50.0
MAX_SPEED_MPS = 80.0 / 3.6
TRIP_RESET_M = 1_000.0
MIN_VALID_STOPS_PER_TRIP = 5

OFFSET_SCALE_M = 30.0
BACKWARD_SCALE_M = 50.0
MAX_BACKWARD_COST = 3.0
SWITCH_COST = 8.0

PROJECTION_CHUNK = 20_000


@dataclass(frozen=True)
class Route:
    """Ordered stops of one line in one direction, as a polyline in meters."""

    line: str
    direction: str
    stop_names: list[str]
    districts: list[str]
    xy: np.ndarray
    stop_s: np.ndarray


@dataclass(frozen=True)
class Origin:
    """Reference point of the local equirectangular projection."""

    lat: float
    lon: float


# --------------------------------------------------------------------------- #
# 1. Routes
# --------------------------------------------------------------------------- #


def to_xy(lat: np.ndarray, lon: np.ndarray, origin: Origin) -> np.ndarray:
    """Project lat/lon (degrees) to local planar meters around ``origin``.

    Args:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        origin: Projection center.

    Returns:
        Array of shape (n, 2) with x (east) and y (north) in meters.
    """
    lon_scale = METERS_PER_DEG_LON_AT_EQUATOR * np.cos(np.radians(origin.lat))
    x = (lon - origin.lon) * lon_scale
    y = (lat - origin.lat) * METERS_PER_DEG_LAT
    return np.column_stack([x, y])


def load_routes(stops_path: Path, line: str) -> tuple[list[Route], Origin]:
    """Build one route polyline per direction of ``line``.

    Directions with fewer than ``MIN_STOPS_PER_DIRECTION`` stops are ignored
    (they are short variants such as "via Freguesia").

    Args:
        stops_path: Pipe-separated stops file (file order == route order).
        line: Bus line number, e.g. "415".

    Returns:
        The routes and the projection origin shared by routes and pings.

    Raises:
        FileNotFoundError: If the stops file does not exist.
        ValueError: If the line has no direction with enough stops.
    """
    stops = pd.read_csv(stops_path, sep="|", dtype={"line_number": str})
    stops = stops[stops["line_number"] == line]
    if stops.empty:
        raise ValueError(f"line {line} not found in {stops_path}")

    origin = Origin(
        lat=float(stops["stop_latitude"].mean()),
        lon=float(stops["stop_longitude"].mean()),
    )
    routes: list[Route] = []
    for direction, group in stops.groupby("direction", sort=False):
        if len(group) < MIN_STOPS_PER_DIRECTION:
            continue
        xy = to_xy(
            group["stop_latitude"].to_numpy(),
            group["stop_longitude"].to_numpy(),
            origin,
        )
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        routes.append(
            Route(
                line=line,
                direction=str(direction),
                stop_names=group["stop_name"].astype(str).tolist(),
                districts=group["district"].astype(str).tolist(),
                xy=xy,
                stop_s=np.concatenate([[0.0], np.cumsum(step)]),
            )
        )
    if not routes:
        raise ValueError(f"line {line} has no direction with enough stops")
    return routes, origin


# --------------------------------------------------------------------------- #
# 2. Projection
# --------------------------------------------------------------------------- #


def project_onto_route(
    points: np.ndarray, route: Route
) -> tuple[np.ndarray, np.ndarray]:
    """Project planar points onto the route polyline.

    Args:
        points: Array (n, 2) in meters.
        route: Target route.

    Returns:
        ``s`` (along-route position, meters) and ``offset`` (lateral distance,
        meters), both of shape (n,).
    """
    start = route.xy[:-1]
    vector = route.xy[1:] - route.xy[:-1]
    length_sq = np.maximum((vector**2).sum(axis=1), 1e-9)
    length = np.sqrt(length_sq)

    s_out = np.empty(len(points))
    offset_out = np.empty(len(points))
    for begin in range(0, len(points), PROJECTION_CHUNK):
        chunk = points[begin : begin + PROJECTION_CHUNK]
        relative = chunk[:, None, :] - start[None, :, :]
        fraction = np.clip((relative * vector).sum(axis=2) / length_sq, 0.0, 1.0)
        nearest = start[None, :, :] + fraction[:, :, None] * vector[None, :, :]
        distance = np.linalg.norm(chunk[:, None, :] - nearest, axis=2)

        best = distance.argmin(axis=1)
        rows = np.arange(len(chunk))
        s_out[begin : begin + len(chunk)] = (
            route.stop_s[best] + fraction[rows, best] * length[best]
        )
        offset_out[begin : begin + len(chunk)] = distance[rows, best]
    return s_out, offset_out


# --------------------------------------------------------------------------- #
# 3. Direction inference and trips
# --------------------------------------------------------------------------- #


def emission_costs(s: np.ndarray, offset: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cost of assigning each ping to each direction.

    Args:
        s: Along-route positions, shape (n, k) for k directions.
        offset: Lateral offsets, shape (n, k).
        t: Ping times in seconds, shape (n,).

    Returns:
        Cost matrix (n, k); lower is more plausible.
    """
    cost = offset / OFFSET_SCALE_M
    backward = np.zeros_like(s)
    backward[1:] = np.maximum(0.0, -(s[1:] - s[:-1])) / BACKWARD_SCALE_M
    same_run = np.zeros(len(t), dtype=bool)
    same_run[1:] = np.diff(t) <= MAX_TRIP_GAP_S
    backward[~same_run] = 0.0
    return np.asarray(cost + np.minimum(backward, MAX_BACKWARD_COST), dtype=float)


def viterbi(cost: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Most plausible direction sequence for one vehicle.

    Args:
        cost: Emission costs (n, k).
        t: Ping times in seconds (n,). Switching is free across long gaps.

    Returns:
        Index of the chosen direction for every ping, shape (n,).
    """
    n, k = cost.shape
    total = cost[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        switch = 0.0 if t[i] - t[i - 1] > MAX_TRIP_GAP_S else SWITCH_COST
        transition = total[:, None] + switch * (1 - np.eye(k))
        back[i] = transition.argmin(axis=0)
        total = transition.min(axis=0) + cost[i]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(total.argmin())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def split_trips(direction: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Trip number for every ping of one vehicle.

    A new trip starts when the direction changes, when there is a gap longer
    than ``MAX_TRIP_GAP_S`` or when the position resets backward by more than
    ``TRIP_RESET_M`` (bus restarted the route).

    Args:
        direction: Direction index per ping.
        s: Along-route position in the chosen direction.
        t: Ping times in seconds.

    Returns:
        Trip number per ping (0, 1, 2, ...).
    """
    trip = np.zeros(len(t), dtype=np.int64)
    current = 0
    running_max = s[0] if len(s) else 0.0
    for i in range(1, len(t)):
        changed = direction[i] != direction[i - 1]
        gap = t[i] - t[i - 1] > MAX_TRIP_GAP_S
        reset = s[i] < running_max - TRIP_RESET_M
        if changed or gap or reset:
            current += 1
            running_max = s[i]
        running_max = max(running_max, s[i])
        trip[i] = current
    return trip


# --------------------------------------------------------------------------- #
# 4. Cleaning
# --------------------------------------------------------------------------- #


def clean_trip(s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop impossible forward jumps and make ``s`` non-decreasing.

    Args:
        s: Along-route positions of one trip, ordered by time.
        t: Ping times in seconds.

    Returns:
        Cleaned ``s`` (non-decreasing) and the matching ``t``.
    """
    keep = [0]
    for i in range(1, len(s)):
        last = keep[-1]
        elapsed = max(t[i] - t[last], 1.0)
        if (s[i] - s[last]) / elapsed <= MAX_SPEED_MPS:
            keep.append(i)
    kept = np.asarray(keep)
    s_clean = np.maximum.accumulate(s[kept])
    t_clean = t[kept]
    first, last = moving_window(s_clean)
    return s_clean[first : last + 1], t_clean[first : last + 1]


def moving_window(s: np.ndarray) -> tuple[int, int]:
    """Indices that trim terminal layovers (bus parked at start or end).

    Keeps the last ping before the bus starts moving and the first ping after
    it stops for good, so stationary periods do not distort interpolation.

    Args:
        s: Non-decreasing along-route positions.

    Returns:
        First and last index (inclusive) of the moving part of the trip.
    """
    moved = np.flatnonzero(s > s[0] + STATIONARY_TOLERANCE_M)
    first = max(int(moved[0]) - 1, 0) if moved.size else 0
    before_end = np.flatnonzero(s < s[-1] - STATIONARY_TOLERANCE_M)
    last = int(before_end[-1]) + 1 if before_end.size else len(s) - 1
    return first, last


# --------------------------------------------------------------------------- #
# 5. Interpolation and segments
# --------------------------------------------------------------------------- #


def interpolate_times(
    s: np.ndarray, t: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Passage time at each target position, from the two bracketing pings.

    Args:
        s: Non-decreasing along-route positions of one trip.
        t: Ping times in seconds.
        targets: Positions to time (e.g. stop positions).

    Returns:
        Interpolated times and the bracketing gap in seconds. Both are NaN when
        the target is outside the trip or the gap exceeds ``MAX_BRACKET_GAP_S``.
    """
    after = np.searchsorted(s, targets, side="left")
    inside = (after > 0) & (after < len(s))
    before = np.clip(after - 1, 0, len(s) - 1)
    after = np.clip(after, 0, len(s) - 1)

    gap = t[after] - t[before]
    distance = np.maximum(s[after] - s[before], 1e-9)
    fraction = (targets - s[before]) / distance
    times = t[before] + fraction * gap

    valid = inside & (gap <= MAX_BRACKET_GAP_S)
    return np.where(valid, times, np.nan), np.where(inside, gap, np.nan)


def trip_segments(
    route: Route, s: np.ndarray, t: np.ndarray
) -> tuple[pd.DataFrame, int, int]:
    """Segments (stop k -> k+1) of one trip.

    Args:
        route: Route of the trip direction.
        s: Cleaned positions of the trip.
        t: Ping times in seconds.

    Returns:
        Valid segments, the number of segments expected inside the trip range
        and the number of stops with a valid time.
    """
    stop_times, gaps = interpolate_times(s, t, route.stop_s)
    in_range = (route.stop_s > s[0]) & (route.stop_s <= s[-1])
    expected = int(np.sum(in_range[:-1] & in_range[1:]))

    valid = ~np.isnan(stop_times[:-1]) & ~np.isnan(stop_times[1:])
    index = np.flatnonzero(valid)
    frame = pd.DataFrame(
        {
            "stop_seq": index + 1,
            "stop_from": [route.stop_names[i] for i in index],
            "stop_to": [route.stop_names[i + 1] for i in index],
            "district_from": [route.districts[i] for i in index],
            "distance_m": np.diff(route.stop_s)[index],
            "departure_epoch": stop_times[index],
            "travel_time_s": stop_times[index + 1] - stop_times[index],
            "max_gap_s": np.maximum(gaps[index], gaps[index + 1]),
        }
    )
    return frame, expected, int(np.sum(~np.isnan(stop_times)))


# --------------------------------------------------------------------------- #
# 6. Validation
# --------------------------------------------------------------------------- #


def holdout_errors(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Hide every other ping and predict its time from the remaining ones.

    This is conservative: the remaining pings are ~2x sparser than reality.

    Args:
        s: Cleaned positions of one trip.
        t: Ping times in seconds.

    Returns:
        Absolute errors in seconds for the hidden pings that could be predicted.
    """
    if len(s) < 4:
        return np.empty(0)
    kept_s, kept_t = s[::2], t[::2]
    hidden_s, hidden_t = s[1::2], t[1::2]
    predicted, _ = interpolate_times(kept_s, kept_t, hidden_s)
    usable = ~np.isnan(predicted)
    return np.asarray(np.abs(predicted[usable] - hidden_t[usable]), dtype=float)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_pings(gps_path: Path, line: str, start: str, end: str) -> pd.DataFrame:
    """Read the pings of one line between two dates (inclusive).

    Args:
        gps_path: Parquet with columns ts, veic, linha, lat, lon.
        line: Bus line number.
        start: First service date (YYYY-MM-DD).
        end: Last service date (YYYY-MM-DD).

    Returns:
        Pings ordered by vehicle and time, with an ``epoch`` column in seconds.

    Raises:
        duckdb.IOException: If the parquet file cannot be read.
    """
    query = f"""
        SELECT ts, veic AS vehicle, lat, lon
        FROM read_parquet('{gps_path}')
        WHERE linha = ? AND ts >= CAST(? AS TIMESTAMP)
          AND ts < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY
          AND isfinite(lat) AND isfinite(lon) AND ts IS NOT NULL
        ORDER BY vehicle, ts
    """
    pings = duckdb.execute(query, [line, start, end]).df()
    unix_start = pd.Timestamp("1970-01-01")
    pings["epoch"] = (pings["ts"] - unix_start).dt.total_seconds()
    return pings


@dataclass
class Report:
    """Validation counters accumulated over all trips."""

    pings_total: int = 0
    pings_on_route: int = 0
    trips: int = 0
    trips_used: int = 0
    segments_expected: int = 0
    segments_valid: int = 0


def build_segments(
    line: str, start: str, end: str
) -> tuple[pd.DataFrame, Report, np.ndarray]:
    """Run the full pipeline for one line and date range.

    Args:
        line: Bus line number.
        start: First service date (YYYY-MM-DD).
        end: Last service date (YYYY-MM-DD).

    Returns:
        Segments table, validation counters and holdout absolute errors.
    """
    routes, origin = load_routes(STOPS_PATH, line)
    pings = load_pings(GPS_PATH, line, start, end)
    report = Report(pings_total=len(pings))
    log.info("pings_loaded", line=line, pings=len(pings), routes=len(routes))

    points = to_xy(pings["lat"].to_numpy(), pings["lon"].to_numpy(), origin)
    projections = [project_onto_route(points, route) for route in routes]
    s_all = np.column_stack([p[0] for p in projections])
    offset_all = np.column_stack([p[1] for p in projections])

    on_route = offset_all.min(axis=1) <= MAX_OFFSET_M
    report.pings_on_route = int(on_route.sum())

    frames: list[pd.DataFrame] = []
    errors: list[np.ndarray] = []
    vehicles = pings["vehicle"].to_numpy()
    epochs = pings["epoch"].to_numpy()

    for vehicle in np.unique(vehicles[on_route]):
        mask = on_route & (vehicles == vehicle)
        t = epochs[mask]
        s_vehicle, offset_vehicle = s_all[mask], offset_all[mask]

        direction = viterbi(emission_costs(s_vehicle, offset_vehicle, t), t)
        rows = np.arange(len(t))
        s_chosen = s_vehicle[rows, direction]
        trip_ids = split_trips(direction, s_chosen, t)

        for trip in np.unique(trip_ids):
            in_trip = trip_ids == trip
            report.trips += 1
            route = routes[int(direction[in_trip][0])]
            s_clean, t_clean = clean_trip(s_chosen[in_trip], t[in_trip])
            if len(s_clean) < 2:
                continue
            segments, expected, valid_stops = trip_segments(route, s_clean, t_clean)
            if valid_stops < MIN_VALID_STOPS_PER_TRIP:
                continue

            report.trips_used += 1
            report.segments_expected += expected
            report.segments_valid += len(segments)
            errors.append(holdout_errors(s_clean, t_clean))

            segments.insert(0, "line", line)
            segments.insert(1, "direction", route.direction)
            segments.insert(2, "vehicle", vehicle)
            segments.insert(3, "trip_id", f"{vehicle}-{int(t_clean[0])}")
            frames.append(segments)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not table.empty:
        departure = pd.to_datetime(table["departure_epoch"], unit="s")
        table.insert(4, "departure_ts", departure)
        table.insert(5, "service_date", departure.dt.date)
        table = table.drop(columns="departure_epoch")
    all_errors = np.concatenate(errors) if errors else np.empty(0)
    return table, report, all_errors


def summarize(
    table: pd.DataFrame, report: Report, errors: np.ndarray
) -> dict[str, object]:
    """Validation summary used to approve (or reject) the dataset.

    Args:
        table: Segments table.
        report: Counters accumulated by ``build_segments``.
        errors: Holdout absolute errors in seconds.

    Returns:
        JSON-serializable summary.
    """
    speed_kmh = table["distance_m"] / table["travel_time_s"].clip(lower=1) * 3.6
    coverage = report.segments_valid / max(report.segments_expected, 1)
    holdout_mae = float(errors.mean()) if errors.size else float("nan")
    return {
        "pings_total": report.pings_total,
        "pings_on_route_pct": round(
            100 * report.pings_on_route / max(report.pings_total, 1), 1
        ),
        "trips_detected": report.trips,
        "trips_used": report.trips_used,
        "segments": len(table),
        "segment_coverage_pct": round(100 * coverage, 1),
        "holdout_mae_s": round(holdout_mae, 1),
        "holdout_median_s": round(float(np.median(errors)), 1) if errors.size else None,
        "holdout_p90_s": round(float(np.quantile(errors, 0.9)), 1)
        if errors.size
        else None,
        "holdout_n": int(errors.size),
        "travel_time_s_p10_median_p90": [
            round(float(q), 1) for q in table["travel_time_s"].quantile([0.1, 0.5, 0.9])
        ],
        "speed_in_2_80_kmh_pct": round(100 * float(speed_kmh.between(2, 80).mean()), 1),
        "non_positive_travel_time": int((table["travel_time_s"] <= 0).sum()),
        "trips_per_day": table.groupby("service_date")["trip_id"]
        .nunique()
        .describe()
        .round(1)
        .to_dict(),
        "approved": bool(coverage >= 0.70 and holdout_mae <= 30.0),
    }


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--line", default="415")
    parser.add_argument("--start", default="2019-02-11")
    parser.add_argument("--end", default="2019-02-17")
    parser.add_argument("--tag", default=None, help="Output name suffix.")
    args = parser.parse_args()

    table, report, errors = build_segments(args.line, args.start, args.end)
    summary = summarize(table, report, errors)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.line}_{args.start}_{args.end}"
    table.to_parquet(OUTPUT_DIR / f"segments_{tag}.parquet", index=False)
    (OUTPUT_DIR / f"validation_{tag}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("segments_written", tag=tag, rows=len(table))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
