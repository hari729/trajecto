from pathlib import Path

import numpy as np
import pytest

from trajecto.problem import TrajectoryProblem, bspline_trajecto


URDF_PATH = Path(__file__).parent / "ur5.urdf"
WAYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.57, -1.57, 1.57, -1.57, 1.57, -1.57],
        [3.14, -3.14, 3.14, -3.14, 3.14, -3.14],
        [4.71, -4.71, 4.71, -4.71, 4.71, -4.71],
    ]
)


@pytest.fixture
def trajectory_extras():
    return {
        "waypoints_cont": WAYPOINTS,
        "end_v": np.zeros(6),
    }


@pytest.fixture
def problem(trajectory_extras):
    return TrajectoryProblem(
        trajectory_function=bspline_trajecto,
        urdf_path=str(URDF_PATH),
        n_var=3,
        bounds=np.array([[0.5, 1.0, 1.0], [10.0, 10.0, 10.0]]),
        trajectory_extras=trajectory_extras,
        joint_limits={
            "dq": np.full(6, 2 * np.pi),
            "ddq": np.full(6, 10.0),
            "dddq": np.full(6, 50.0),
            "tau": np.full(6, 100.0),
        },
        time_limit=10.0,
    )


class TestTrajectoryProblem:
    def test_bspline_returns_valid_time_and_joint_array_shapes(self, trajectory_extras):
        trajectory = bspline_trajecto(np.array([3.3, 3.3, 3.3]), trajectory_extras)

        assert trajectory["t"].shape == (500,)
        assert trajectory["t"][0] == pytest.approx(0.0)
        assert trajectory["t"][-1] == pytest.approx(9.9)
        assert np.all(np.diff(trajectory["t"]) > 0.0)

        for key in ("q", "dq", "ddq", "dddq"):
            assert trajectory[key].shape == (500, 6)
            assert np.all(np.isfinite(trajectory[key]))

    def test_problem_metadata_matches_fixed_objectives_and_constraints(self, problem):
        assert problem.n_obj == 3
        assert problem.n_ieq_constr == 4 * problem.pin_model.nv + 1
        assert problem.n_ieq_constr == 25

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
