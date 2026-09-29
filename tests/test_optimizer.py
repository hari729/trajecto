from pathlib import Path

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize

from trajecto.optimizer import (
    find_knee_point,
    merge_pareto_fronts,
    save_trajectory_results,
)
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

TRAJECTORY_EXTRAS = {
    "waypoints": WAYPOINTS,
    "bconditions": BCONDITIONS,
    "num_joints": 6,
    "k": 6,
    "steps": 500,
}

BOUNDS = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])


@pytest.fixture
def problem():
    return TrajectoryProblem(
        trajectory_function=bspline_trajectory,
        urdf_arg={"source": str(URDF_PATH)},
        n_var=3,
        bounds=BOUNDS,
        trajectory_extras=TRAJECTORY_EXTRAS,
        joint_limits=JOINT_LIMITS,
        time_limit=10.0,
        n_threads=1,
    )


class _Result:
    """Minimal stand-in for a pymoo Result used by merge_pareto_fronts."""

    def __init__(self, opt):
        self.opt = opt


def _population(F):
    pop = Population.new("F", np.asarray(F, dtype=float))
    pop.set("X", np.zeros((len(pop), 3)))
    pop.set("G", np.zeros((len(pop), 25)))
    return pop


class TestMergeParetoFronts:
    def test_empty_results_return_empty_population(self):
        assert len(merge_pareto_fronts([])) == 0
        assert len(merge_pareto_fronts([_Result(None)])) == 0
        assert len(merge_pareto_fronts([_Result(Population.empty())])) == 0

    def test_dominating_point_does_not_survive(self):
        # every objective of point b is worse than point a
        pop = _population([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        merged = merge_pareto_fronts([_Result(pop)])
        np.testing.assert_allclose(merged.get("F"), [[1.0, 1.0, 1.0]])

    def test_non_dominated_union_across_results(self):
        pop_a = _population([[1.0, 2.0, 2.0]])
        pop_b = _population([[2.0, 1.0, 2.0]])
        merged = merge_pareto_fronts([_Result(pop_a), _Result(pop_b)])
        assert len(merged) == 2

    def test_cross_result_domination_is_removed(self):
        pop_a = _population([[1.0, 1.0, 1.0]])
        pop_b = _population([[2.0, 2.0, 2.0]])
        merged = merge_pareto_fronts([_Result(pop_a), _Result(pop_b)])
        np.testing.assert_allclose(merged.get("F"), [[1.0, 1.0, 1.0]])


class TestFindKneePoint:
    def test_middle_of_front_is_the_knee(self):
        # normalized objective space: [0,1,1] and [1,1,0] are extreme corners,
        # [0.2, 0.2, 0.2] sits closest to the utopia point
        F = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.2, 0.2, 0.2],
                [1.0, 1.0, 0.0],
            ]
        )
        assert find_knee_point(F) == 1

    def test_degenerate_single_objective_range_is_safe(self):
        # constant first objective: normalization must not produce NaN, and the
        # point dominating the other two objectives is the knee
        F = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert find_knee_point(F) == 0


class TestSaveTrajectoryResults:
    def test_writes_csv_and_four_trajectory_jsons(self, problem, tmp_path):
        X = np.array(
            [
                [1.0, 1.0, 1.0],  # fast
                [9.0, 9.0, 9.0],  # slow
            ]
        )
        F, G = problem.evaluate(X, return_values_of=["F", "G"])
        front = Population.new("X", X)
        front.set("F", F)
        front.set("G", G)

        save_trajectory_results(front, tmp_path, problem)

        assert (tmp_path / "trajectory-optimization-results.csv").is_file()
        for name in ("fastest", "efficient", "smoothest", "knee"):
            assert (tmp_path / f"{name}-trajectory.json").is_file()

        import json

        with open(tmp_path / "fastest-trajectory.json") as f:
            fastest = json.load(f)
        assert fastest["joint_names"] == problem.joint_names
        assert len(fastest["position"]) == 500
        assert len(fastest["torque"][0]) == 6


class TestOptimizationSmoke:
    def test_tiny_optimization_run_returns_non_empty_front(self, problem):
        res = minimize(
            problem,
            NSGA2(pop_size=4),
            termination=("n_gen", 4),
            seed=1,
            verbose=False,
        )
        assert res.opt is not None
        assert len(res.opt) > 0
        F = res.opt.get("F")
        G = res.opt.get("G")
        assert F.shape[1] == 3
        assert G.shape[1] == 4 * problem.n_joints + 1
        # durations stay within the variable bounds
        durations = F[:, 0]
        assert np.all(durations >= BOUNDS[0].sum() - 1e-6)
        assert np.all(durations <= BOUNDS[1].sum() + 1e-6)
