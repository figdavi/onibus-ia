"""Unit tests for the shared preprocessing (synthetic data only)."""

import numpy as np
import pandas as pd

import prep


def synthetic_segments() -> pd.DataFrame:
    """Two segments observed every day from 25/01 to 21/03 at 08:00."""
    days = pd.date_range("2019-01-25", "2019-03-21", freq="D")
    rows = []
    for day in days:
        for seq in (1, 2):
            rows.append(
                {
                    "line": "415",
                    "direction": "Leblon",
                    "stop_seq": seq,
                    "departure_ts": day + pd.Timedelta(hours=8),
                    "service_date": day.date(),
                    "travel_time_s": 60.0,
                    "max_gap_s": 60.0,
                }
            )
    return pd.DataFrame(rows)


def test_calendar_flags_carnival_and_weekend() -> None:
    out = prep.add_calendar(synthetic_segments())
    by_day = out.drop_duplicates("service_date").set_index("service_date")
    assert by_day.loc[pd.Timestamp("2019-03-05").date(), "day_type"] == "carnaval"
    assert by_day.loc[pd.Timestamp("2019-02-09").date(), "day_type"] == "sábado"
    assert by_day.loc[pd.Timestamp("2019-02-10").date(), "day_type"] == "domingo"
    assert by_day.loc[pd.Timestamp("2019-02-11").date(), "day_type"] == "útil"
    assert (out["hour"] == 8).all()


def test_temporal_split_has_no_overlap_and_skips_buffer_days() -> None:
    train, val, test = prep.temporal_split(synthetic_segments())
    assert train["service_date"].max() < val["service_date"].min()
    assert val["service_date"].max() < test["service_date"].min()
    used = (
        set(train["service_date"])
        | set(val["service_date"])
        | set(test["service_date"])
    )
    for skipped in ("2019-03-08", "2019-03-10", "2019-03-21"):
        assert pd.Timestamp(skipped).date() not in used


def test_quality_filter_drops_large_gaps() -> None:
    segments = synthetic_segments()
    segments.loc[0, "max_gap_s"] = prep.MAX_GAP_S + 1
    assert len(prep.filter_quality(segments)) == len(segments) - 1


def test_outlier_bounds_come_from_train_and_drop_rare_segments() -> None:
    rng = np.random.default_rng(0)
    train = synthetic_segments().iloc[:80].copy()
    train["travel_time_s"] = rng.normal(60, 5, len(train))
    train = pd.concat([train, train.iloc[[0]].assign(stop_seq=3)])
    bounds = prep.fit_outlier_bounds(train)
    assert set(bounds["stop_seq"]) == {1, 2}

    test = train.iloc[:3].copy()
    test["travel_time_s"] = [60.0, 10_000.0, 60.0]
    kept = prep.apply_outlier_bounds(test, bounds)
    assert list(kept["travel_time_s"]) == [60.0, 60.0]
