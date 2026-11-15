"""Unit tests for the geometric and interpolation steps (synthetic data only)."""

import numpy as np

import rio_segments as rs


def straight_route() -> rs.Route:
    """Route along the x axis with stops every 100 m."""
    xy = np.column_stack([np.arange(0.0, 500.0, 100.0), np.zeros(5)])
    return rs.Route(
        line="000",
        direction="test",
        stop_names=[f"stop {i}" for i in range(5)],
        districts=["synthetic"] * 5,
        xy=xy,
        stop_s=xy[:, 0].copy(),
    )


def test_projection_returns_position_and_offset() -> None:
    points = np.array([[150.0, 20.0], [380.0, -5.0], [450.0, 0.0]])
    s, offset = rs.project_onto_route(points, straight_route())
    np.testing.assert_allclose(s, [150.0, 380.0, 400.0])
    np.testing.assert_allclose(offset, [20.0, 5.0, 50.0])


def test_interpolation_is_linear_between_bracketing_pings() -> None:
    s = np.array([0.0, 200.0, 400.0])
    t = np.array([0.0, 60.0, 120.0])
    times, gaps = rs.interpolate_times(s, t, np.array([100.0, 300.0]))
    np.testing.assert_allclose(times, [30.0, 90.0])
    np.testing.assert_allclose(gaps, [60.0, 60.0])


def test_interpolation_rejects_long_gaps_and_out_of_range() -> None:
    s = np.array([0.0, 200.0])
    t = np.array([0.0, rs.MAX_BRACKET_GAP_S + 1])
    times, _ = rs.interpolate_times(s, t, np.array([100.0, 500.0]))
    assert np.isnan(times).all()


def test_clean_trip_drops_impossible_jump_and_terminal_layover() -> None:
    s = np.array([0.0, 0.0, 0.0, 300.0, 5_000.0, 600.0])
    t = np.array([0.0, 60.0, 120.0, 180.0, 190.0, 240.0])
    s_clean, t_clean = rs.clean_trip(s, t)
    assert 5_000.0 not in s_clean
    assert t_clean[0] == 120.0
    assert (np.diff(s_clean) >= 0).all()
