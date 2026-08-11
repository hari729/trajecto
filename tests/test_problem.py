from pathlib import Path

import numpy as np
import pytest

from trajecto.problem import TrajectoryProblem
from trajecto.samples import bspline_trajectory

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
}


@pytest.fixture
def trajectory_extras():
    return {
        "waypoints": WAYPOINTS,
        "bconditions": BCONDITIONS,
        "num_joints": 6,
        "k": 6,
        "steps": 500,
    }


@pytest.fixture
def problem(trajectory_extras):
    return TrajectoryProblem(
        trajectory_function=bspline_trajectory,
        urdf_arg={"source": str(URDF_PATH)},
        n_var=3,
        bounds=np.array([[0.5, 1.0, 1.0], [10.0, 10.0, 10.0]]),
        trajectory_extras=trajectory_extras,
        joint_limits=JOINT_LIMITS,
        time_limit=10.0,
    )


class TestTrajectoryProblem:
    def test_bspline_returns_valid_time_and_joint_array_shapes(self, trajectory_extras):
        trajectory = bspline_trajectory(np.array([3.3, 3.3, 3.3]), **trajectory_extras)

        assert trajectory["time"].shape == (500,)
        assert trajectory["time"][0] == pytest.approx(0.0)
        assert trajectory["time"][-1] == pytest.approx(9.9)
        assert np.all(np.diff(trajectory["time"]) > 0.0)

        for key in ("position", "velocity", "acceleration", "jerk"):
            assert trajectory[key].shape == (500, 6)
            assert np.all(np.isfinite(trajectory[key]))

    def test_problem_metadata_matches_fixed_objectives_and_constraints(self, problem):
        assert problem.n_obj == 3
        assert problem.n_ieq_constr == 4 * problem.pin_model.nv + 1
        assert problem.n_ieq_constr == 25

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

    def test_trajectory_columns_follow_model_joint_order(
        self, problem, trajectory_extras
    ):
        trajectory = bspline_trajectory(np.array([3.3, 3.3, 3.3]), **trajectory_extras)

        np.testing.assert_allclose(trajectory["position"][0], WAYPOINTS[0], atol=1e-9)
        np.testing.assert_allclose(trajectory["position"][-1], WAYPOINTS[-1], atol=1e-9)

    def test_single_evaluation_returns_expected_shapes_and_duration(self, problem):
        f, g = problem.evaluate(
            np.array([[3.3, 3.3, 3.3]]), return_values_of=["F", "G"]
        )

        assert f.shape == (1, 3)
        assert g.shape == (1, 25)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(g))
        assert f[0, 0] == pytest.approx(9.9)
        assert g[0, 0] == pytest.approx(-0.1)

    def test_batch_evaluation_returns_one_result_per_candidate(self, problem):
        x = np.array([[3.3, 3.3, 3.3], [2.5, 3.2, 3.8]])
        f, g = problem.evaluate(x, return_values_of=["F", "G"])

        assert f.shape == (2, 3)
        assert g.shape == (2, 25)
        assert np.all(np.isfinite(f))
        assert np.all(np.isfinite(g))
        np.testing.assert_allclose(f[:, 0], [9.9, 9.5])
        np.testing.assert_allclose(g[:, 0], [-0.1, -0.5])

    def test_generate_trajectory_stamps_joint_names_and_torque(self, problem):
        trajectory = problem.generate_trajectory(np.array([3.3, 3.3, 3.3]))

        assert trajectory["joint_names"] == problem.joint_names
        assert trajectory["time"].shape == (500,)
        assert np.asarray(trajectory["torque"]).shape == (500, 6)
        assert np.all(np.isfinite(trajectory["torque"]))
