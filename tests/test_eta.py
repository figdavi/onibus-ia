"""Unit tests for the chained ETA evaluation (synthetic data only)."""

import numpy as np
import pandas as pd

import common
import eta


def synthetic_trips(n_buses: int = 4, n_segments: int = 6) -> pd.DataFrame:
    """Buses 10 min apart, every segment takes 60 s."""
    base = pd.Timestamp("2019-03-11 08:00")
    rows = []
    for bus in range(n_buses):
        start = base + pd.Timedelta(minutes=10 * bus)
        for seq in range(2, 2 + n_segments):
            rows.append(
                {
                    "trip_id": f"T{bus}",
                    "direction": "Usina",
                    "stop_seq": seq,
                    "stop_to": f"stop {seq + 1}",
                    "departure_ts": start + pd.Timedelta(seconds=60 * (seq - 2)),
                    "distance_m": 300.0,
                    "travel_time_s": 60.0,
                }
            )
    return pd.DataFrame(rows)


class Constant:
    """Predicts the same travel time for every segment."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.seconds)


def test_chains_require_consecutive_segments() -> None:
    segments = synthetic_trips(n_buses=1, n_segments=4)
    segments = segments[segments["stop_seq"] != 4]  # gap after stop_seq 3
    enriched = common.enrich(segments, segments).reset_index(drop=True)
    starts = enriched.iloc[eta.chains(enriched, 2)[:, 0]]["stop_seq"].tolist()
    assert starts == [2]


def test_evaluate_eta_sums_segment_errors() -> None:
    segments = synthetic_trips()
    table = eta.evaluate_eta(
        {"exact": Constant(60.0), "slow": Constant(70.0)},
        segments,
        reference=segments,
        horizons=(1, 3),
    )
    mae = table.set_index(["model", "stops_ahead"])["MAE_s"]
    assert mae[("exact", 3)] == 0.0
    assert mae[("slow", 1)] == 10.0
    assert mae[("slow", 3)] == 30.0


def test_replay_returns_one_row_per_stop() -> None:
    segments = synthetic_trips()
    result = eta.replay(
        {"exact": Constant(60.0)}, segments, segments, "T1", origin_seq=2, horizon=3
    )
    assert result["stop_seq"].tolist() == [3, 4, 5]
    assert (result["exact"] == result["real"]).all()
