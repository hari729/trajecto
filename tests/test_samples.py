import numpy as np
import pytest

from trajecto.samples import BSplineTrajectory, Trajectory, bspline_trajectory

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


class TestBSplineTrajectoryClass:
    def test_defaults_n_var_and_bounds_from_waypoints(self):
        t = BSplineTrajectory(waypoints=WAYPOINTS, k=6, steps=100)
        assert t.n_var == WAYPOINTS.shape[0] - 1  # one per segment
        assert t.bounds.shape == (2, t.n_var)
        assert np.all(t.bounds[0] < t.bounds[1])
        assert t.n_joints == 6

    def test_explicit_clamped_bconditions_clamp_start(self):
        clamped = [
            ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)])
        ] * 6
        t = BSplineTrajectory(
            waypoints=WAYPOINTS, k=6, steps=100, bconditions=clamped
        )
        traj = t(np.full(t.n_var, 2.0))
        # zero velocity/acceleration at the clamped start
        np.testing.assert_allclose(traj["velocity"][0], 0.0, atol=1e-8)
        np.testing.assert_allclose(traj["acceleration"][0], 0.0, atol=1e-8)
        np.testing.assert_allclose(traj["position"][0], WAYPOINTS[0], atol=1e-9)

    def test_returns_validated_dict(self):
        t = BSplineTrajectory(waypoints=WAYPOINTS, k=3, steps=50)
        traj = t(np.full(t.n_var, 1.0))
        assert set(traj) == {"time", "position", "velocity", "acceleration", "jerk"}
        assert traj["position"].shape == (50, 6)

    def test_3dof_robot(self):
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [2.0, 1.0, -0.5],
                [1.0, 2.0, 1.0],
            ]
        )
        t = BSplineTrajectory(waypoints=waypoints, k=3, steps=100)
        traj = t(np.full(t.n_var, 1.5))
        assert traj["position"].shape == (100, 3)
        assert t.n_joints == 3
        np.testing.assert_allclose(traj["position"][-1], waypoints[-1], atol=1e-9)


class _BadTrajectory(Trajectory):
    def __init__(self, bad):
        self.bad = bad
        super().__init__(n_var=2, bounds=np.array([[0.0, 0.0], [1.0, 1.0]]))

    def _generate(self, x):
        return self.bad


class TestTrajectoryValidation:
    def test_missing_key_raises(self):
        bad = {
            "time": np.linspace(0, 1, 5),
            "position": np.zeros((5, 2)),
            # velocity/acceleration/jerk missing
        }
        with pytest.raises(ValueError, match="missing keys"):
            _BadTrajectory(bad)(np.array([0.5, 0.5]))

    def test_wrong_shape_raises(self):
        bad = {
            "time": np.linspace(0, 1, 5),
            "position": np.zeros((4, 2)),  # n_steps mismatch
            "velocity": np.zeros((5, 2)),
            "acceleration": np.zeros((5, 2)),
            "jerk": np.zeros((5, 2)),
        }
        with pytest.raises(ValueError, match="shape"):
            _BadTrajectory(bad)(np.array([0.5, 0.5]))

    def test_non_finite_raises(self):
        bad = {
            "time": np.linspace(0, 1, 5),
            "position": np.full((5, 2), np.nan),
            "velocity": np.zeros((5, 2)),
            "acceleration": np.zeros((5, 2)),
            "jerk": np.zeros((5, 2)),
        }
        with pytest.raises(ValueError, match="non-finite"):
            _BadTrajectory(bad)(np.array([0.5, 0.5]))

    def test_non_increasing_time_raises(self):
        bad = {
            "time": np.array([0.0, 0.5, 0.5, 1.0, 2.0]),
            "position": np.zeros((5, 2)),
            "velocity": np.zeros((5, 2)),
            "acceleration": np.zeros((5, 2)),
            "jerk": np.zeros((5, 2)),
        }
        with pytest.raises(ValueError, match="strictly increasing"):
            _BadTrajectory(bad)(np.array([0.5, 0.5]))

    def test_bad_bounds_shape_rejected(self):
        class _T(Trajectory):
            def _generate(self, x):
                return {}

        with pytest.raises(ValueError, match="bounds"):
            _T(n_var=2, bounds=np.array([[0.0], [1.0]]))
