import numpy as np
import pytest

from trajecto.samples import bspline_trajectory

WAYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.57, -1.57, 1.57, -1.57, 1.57, -1.57],
        [3.14, -3.14, 3.14, -3.14, 3.14, -3.14],
        [4.71, -4.71, 4.71, -4.71, 4.71, -4.71],
    ]
)

BCONDITIONS = [
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
]

EXTRAS = {
    "waypoints": WAYPOINTS,
    "bconditions": BCONDITIONS,
    "num_joints": 6,
    "k": 6,
    "steps": 200,
}


class TestBsplineTrajectory:
    def test_contract_keys_and_shapes(self):
        trajectory = bspline_trajectory(np.array([1.0, 2.0, 3.0]), **EXTRAS)

        assert set(trajectory) == {
            "time",
            "position",
            "velocity",
            "acceleration",
            "jerk",
        }
        assert trajectory["time"].shape == (200,)
        for key in ("position", "velocity", "acceleration", "jerk"):
            assert trajectory[key].shape == (200, 6)

    def test_segment_durations_set_total_duration(self):
        x = np.array([1.0, 2.0, 3.0])
        trajectory = bspline_trajectory(x, **EXTRAS)
        assert trajectory["time"][-1] == pytest.approx(x.sum())
        assert trajectory["time"][0] == pytest.approx(0.0)
        assert np.all(np.diff(trajectory["time"]) > 0.0)

    def test_passes_through_waypoints(self):
        trajectory = bspline_trajectory(np.array([1.0, 2.0, 3.0]), **EXTRAS)
        np.testing.assert_allclose(trajectory["position"][0], WAYPOINTS[0], atol=1e-9)
        np.testing.assert_allclose(trajectory["position"][-1], WAYPOINTS[-1], atol=1e-9)

    def test_intermediate_waypoints_hit_at_knot_times(self):
        from scipy.interpolate import make_interp_spline

        x = np.array([1.0, 2.0, 3.0])
        knots = np.concatenate(([0.0], np.cumsum(x)))
        # evaluate the same spline the generator builds, at the exact knot
        # times (the returned grid may not land exactly on knots)
        for j in range(WAYPOINTS.shape[1]):
            spline = make_interp_spline(
                knots, WAYPOINTS[:, j], k=EXTRAS["k"], bc_type=BCONDITIONS[j]
            )
            np.testing.assert_allclose(spline(knots), WAYPOINTS[:, j], atol=1e-9)
