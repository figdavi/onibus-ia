"""Calendar helpers for the temporal EDA notebook.

The cleaning and the split used by every model live in ``common.py``
(``common.prepare`` and ``common.split``). This module only adds the calendar
columns the EDA groups by.

Usage:
    from prep import add_calendar
    segments = add_calendar(pd.read_parquet("segments_415_v0.parquet"))
"""

from __future__ import annotations

import pandas as pd

import common

SEGMENT_KEY = ["line", "direction", "stop_seq"]

WEEKDAY_NAMES = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


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
    service_date = pd.to_datetime(out["service_date"])
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.day_of_week
    out["is_weekend"] = out["weekday"] >= 5
    out["is_carnival"] = service_date.between(
        common.CARNIVAL_START, common.CARNIVAL_END
    )
    day_type = pd.Series("útil", index=out.index)
    day_type[out["weekday"] == 5] = "sábado"
    day_type[out["weekday"] == 6] = "domingo"
    day_type[out["is_carnival"]] = "carnaval"
    out["day_type"] = day_type
    return out
