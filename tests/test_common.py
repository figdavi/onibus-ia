"""Unit tests for shared features and evaluation (synthetic data only)."""

import pandas as pd

import common


def synthetic_segments() -> pd.DataFrame:
    """Two buses on the same two segments, 10 min apart."""
    base = pd.Timestamp("2019-03-04 08:00")
    rows = []
    for bus, start in [("A", base), ("B", base + pd.Timedelta(minutes=10))]:
        for seq, travel in [(1, 60.0), (2, 90.0)]:
            offset = 0 if seq == 1 else 60
            rows.append(
                {
                    "trip_id": f"{bus}-1",
                    "direction": "Leblon",
                    "stop_seq": seq,
                    "departure_ts": start + pd.Timedelta(seconds=offset),
                    "distance_m": 300.0,
                    "travel_time_s": travel,
                }
            )
    return pd.DataFrame(rows)


def test_lags_use_only_past_information() -> None:
    enriched = common.add_lag_features(synthetic_segments()).sort_values(
        ["trip_id", "stop_seq"]
    )
    first_bus = enriched[enriched["trip_id"] == "A-1"]
    second_bus = enriched[enriched["trip_id"] == "B-1"]

    assert pd.isna(first_bus["lag_previous_segment_s"].iloc[0])
    assert first_bus["lag_previous_segment_s"].iloc[1] == 60.0
    assert pd.isna(first_bus["lag_previous_bus_s"]).all()
    assert second_bus["lag_previous_bus_s"].tolist() == [60.0, 90.0]
    assert second_bus["minutes_since_previous_bus"].tolist() == [10.0, 10.0]


def test_calendar_flags_carnival_and_weekday() -> None:
    enriched = common.add_calendar_features(synthetic_segments())
    assert (enriched["is_carnival"] == 1).all()
    assert (enriched["day_of_week"] == 0).all()
    assert (enriched["hour"] == 8).all()


def test_build_features_has_fixed_columns() -> None:
    X, y = common.build_features(synthetic_segments(), reference=synthetic_segments())
    assert {"direction_Leblon", "direction_Usina"} <= set(X.columns)
    assert len(X) == len(y) == 4


def test_baseline_and_evaluate() -> None:
    X, y = common.build_features(synthetic_segments(), reference=synthetic_segments())
    baseline = common.MedianBaseline().fit(X, y)
    metrics = common.evaluate(baseline, X, y)
    assert metrics["MAE_s"] == 0.0
    assert metrics["n"] == 4


def test_prepare_removes_implausible_rows_and_reports() -> None:
    segments = synthetic_segments().assign(service_date="2019-03-04")
    segments.loc[0, "travel_time_s"] = 5.0  # 300 m in 5 s = 216 km/h
    cleaned = common.prepare(segments)
    report = cleaned.attrs["prepare_report"]
    assert len(cleaned) == 3
    assert report["implausible_speed"] == 1
    assert report["rows_in"] - report["rows_out"] == 1


def test_split_is_temporal_and_skips_buffer_days() -> None:
    days = pd.date_range("2019-01-25", "2019-03-21", freq="D")
    table = pd.DataFrame({"service_date": days.date, "travel_time_s": 60.0})

    train, val, test = common.split(table)

    assert train["service_date"].max() == pd.Timestamp("2019-03-03").date()
    assert sorted(val["service_date"]) == list(
        pd.date_range("2019-03-04", "2019-03-07").date
    )
    assert test["service_date"].min() == pd.Timestamp("2019-03-11").date()
    assert test["service_date"].max() == pd.Timestamp("2019-03-20").date()
    assert len(train) + len(val) + len(test) == len(table) - 4


def test_profile_features_use_reference_and_past_only() -> None:
    segments = synthetic_segments()
    reference = segments.assign(travel_time_s=segments["travel_time_s"] * 2)
    enriched = common.add_profile_features(segments, reference).sort_values(
        ["trip_id", "stop_seq"]
    )
    first_bus = enriched[enriched["trip_id"] == "A-1"]

    assert first_bus["segment_median_s"].tolist() == [120.0, 180.0]
    assert pd.isna(first_bus["trip_pace_last"].iloc[0])
    assert first_bus["trip_pace_last"].iloc[1] == 0.5  # 60 s / 120 s
    assert first_bus["trip_pace_mean"].iloc[1] == 0.5


def test_recent_buses_mean_only_counts_arrived_buses() -> None:
    enriched = common.add_lag_features(synthetic_segments())
    second_bus = enriched[enriched["trip_id"] == "B-1"].sort_values("stop_seq")
    first_bus = enriched[enriched["trip_id"] == "A-1"]
    assert second_bus["recent_buses_mean_s"].tolist() == [60.0, 90.0]
    assert first_bus["recent_buses_mean_s"].isna().all()
