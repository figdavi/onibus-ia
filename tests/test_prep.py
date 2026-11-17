"""Unit tests for the EDA calendar helpers (synthetic data only)."""

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
