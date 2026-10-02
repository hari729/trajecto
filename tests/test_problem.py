from pathlib import Path

import numpy as np
import pytest

from trajecto.problem import TrajectoryProblem
from trajecto.samples import BSplineTrajectory, bspline_trajectory

URDF_PATH = Path(__file__).parent / "ur5.urdf"

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

JOINT_LIMITS = {
    "velocity": np.full(6, 2 * np.pi),
    "acceleration": np.full(6, 10.0),
    "jerk": np.full(6, 50.0),
    "torque": np.full(6, 100.0),
    "position": np.array(
        [
            [-6.283185307179586] * 6,
            [6.283185307179586] * 6,
        ],
        dtype=float,
    ),
}

TRAJECTORY_EXTRAS = {
    "waypoints": WAYPOINTS,
    "bconditions": BCONDITIONS,
    "num_joints": 6,
    "k": 6,
    "steps": 500,
}

BOUNDS = np.array([[0.5, 1.0, 1.0], [10.0, 10.0, 10.0]])


@pytest.fixture
def trajectory():
    return BSplineTrajectory(
        waypoints=WAYPOINTS, k=6, steps=500, bconditions=BCONDITIONS, bounds=BOUNDS
    )


@pytest.fixture
def problem(trajectory):
    return TrajectoryProblem(
        trajectory=trajectory,
        urdf_arg={"source": str(URDF_PATH)},
        joint_limits=JOINT_LIMITS,
        time_limit=10.0,
    )


class TestTrajectoryProblem:
    def test_bspline_returns_valid_time_and_joint_array_shapes(self, trajectory):
        traj = trajectory(np.array([3.3, 3.3, 3.3]))

        assert traj["time"].shape == (500,)
        assert traj["time"][0] == pytest.approx(0.0)
        assert traj["time"][-1] == pytest.approx(9.9)
        assert np.all(np.diff(traj["time"]) > 0.0)

        for key in ("position", "velocity", "acceleration", "jerk"):
            assert traj[key].shape == (500, 6)
            assert np.all(np.isfinite(traj[key]))

    def test_problem_metadata_matches_fixed_objectives_and_constraints(self, problem):
        assert problem.n_obj == 3
        assert problem.n_ieq_constr == 6 * problem.pin_model.nv + 1
        assert problem.n_ieq_constr == 37

    def test_joint_names_follow_urdf_declaration_order(self, problem):
        assert problem.joint_names == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        assert len(problem.joint_names) == problem.n_joints

    def test_trajectory_columns_follow_model_joint_order(self, problem, trajectory):
        traj = trajectory(np.array([3.3, 3.3, 3.3]))

        np.testing.assert_allclose(traj["position"][0], WAYPOINTS[0], atol=1e-9)
        np.testing.assert_allclose(traj["position"][-1], WAYPOINTS[-1], atol=1e-9)

    def test_single_evaluation_returns_expected_shapes_and_duration(self, problem):
        f, g = problem.evaluate(
            np.array([[3.3, 3.3, 3.3]]), return_values_of=["F", "G"]
        )

        assert f.shape == (1, 3)
        assert g.shape == (1, 37)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(g))
        assert f[0, 0] == pytest.approx(9.9)
        assert g[0, 0] == pytest.approx(-0.1)

    def test_batch_evaluation_returns_one_result_per_candidate(self, problem):
        x = np.array([[3.3, 3.3, 3.3], [2.5, 3.2, 3.8]])
        f, g = problem.evaluate(x, return_values_of=["F", "G"])

        assert f.shape == (2, 3)
        assert g.shape == (2, 37)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(g))
        np.testing.assert_allclose(f[:, 0], [9.9, 9.5])
        np.testing.assert_allclose(g[:, 0], [-0.1, -0.5])

    def test_joint_limits_derive_velocity_torque_and_position_from_urdf(
        self, trajectory
    ):
        # only acceleration and jerk supplied; velocity/torque/position come from the URDF
        problem = TrajectoryProblem(
            trajectory=trajectory,
            urdf_arg={"source": str(URDF_PATH)},
            joint_limits={"acceleration": np.full(6, 10.0), "jerk": np.full(6, 50.0)},
            time_limit=10.0,
        )
        np.testing.assert_allclose(problem.joint_limits["velocity"], [np.pi] * 6)
        np.testing.assert_allclose(
            problem.joint_limits["torque"], [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
        )
        np.testing.assert_allclose(problem.joint_limits["acceleration"], [10.0] * 6)
        np.testing.assert_allclose(
            problem.joint_limits["position"],
            [
                [-6.283185307179586, -6.283185307179586, -3.141592653589793, -6.283185307179586, -6.283185307179586, -6.283185307179586],
                [6.283185307179586, 6.283185307179586, 3.141592653589793, 6.283185307179586, 6.283185307179586, 6.283185307179586],
            ],
            atol=1e-15,
        )

    def test_wrong_sized_joint_limits_rejected(self, trajectory):
        with pytest.raises(ValueError, match="shape"):
            TrajectoryProblem(
                trajectory=trajectory,
                urdf_arg={"source": str(URDF_PATH)},
                joint_limits={"acceleration": np.full(3, 10.0)},
                time_limit=10.0,
            )

    def test_wrong_sized_position_limit_rejected(self, trajectory):
        with pytest.raises(ValueError, match="position"):
            TrajectoryProblem(
                trajectory=trajectory,
                urdf_arg={"source": str(URDF_PATH)},
                joint_limits={
                    "acceleration": np.full(6, 10.0),
                    "jerk": np.full(6, 50.0),
                    "position": np.zeros((2, 3)),
                },
                time_limit=10.0,
            )

    def test_missing_accel_jerk_raises(self, trajectory):
        with pytest.raises(ValueError, match="acceleration"):
            TrajectoryProblem(
                trajectory=trajectory,
                urdf_arg={"source": str(URDF_PATH)},
                joint_limits=None,
                time_limit=10.0,
            )

    def test_generate_trajectory_stamps_joint_names_and_torque(self, problem):
        trajectory = problem.generate_trajectory(np.array([3.3, 3.3, 3.3]))

        assert trajectory["joint_names"] == problem.joint_names
        assert trajectory["time"].shape == (500,)
        assert np.asarray(trajectory["torque"]).shape == (500, 6)
        assert np.all(np.isfinite(trajectory["torque"]))

    def test_zero_duration_trajectory_is_infeasible_and_finite_free(self, problem):
        def static_trajectory(x, **kwargs):
            t = np.zeros(10)
            return {
                "time": t,
                "position": np.zeros((10, 6)),
                "velocity": np.zeros((10, 6)),
                "acceleration": np.zeros((10, 6)),
                "jerk": np.zeros((10, 6)),
            }

        problem.trajectory_function = static_trajectory
        out = {}
        problem._evaluate(np.array([1.0, 1.0, 1.0]), out)

        assert out["F"][0] == pytest.approx(0.0)
        assert np.isinf(out["F"][1])
        assert np.isinf(out["F"][2])
        assert out["G"][0] == pytest.approx(-problem.time_limit)
        assert len(out["G"]) == 37

    def test_position_limit_violation_gives_positive_constraint(self, problem):
        def bad_trajectory(x, **kwargs):
            t = np.linspace(0, 1.0, 10)
            pos = np.zeros((10, 6))
            pos[:, 0] = np.linspace(0.0, 10.0, 10)  # exceeds upper limit
            return {
                "time": t,
                "position": pos,
                "velocity": np.zeros((10, 6)),
                "acceleration": np.zeros((10, 6)),
                "jerk": np.zeros((10, 6)),
            }

        problem.trajectory_function = bad_trajectory
        out = {}
        problem._evaluate(np.array([1.0, 1.0, 1.0]), out)

        assert out["G"][0] == pytest.approx(1.0 - problem.time_limit)
        pos_lower_start = 1 + 4 * problem.n_joints
        pos_upper_start = 1 + 5 * problem.n_joints
        assert out["G"][pos_upper_start] > 0
        # only the first joint exceeds its upper limit
        np.testing.assert_allclose(
            out["G"][pos_upper_start : pos_upper_start + problem.n_joints],
            [10.0 - 6.283185307179586, -6.283185307179586, -6.283185307179586, -6.283185307179586, -6.283185307179586, -6.283185307179586],
            atol=1e-9,
        )
