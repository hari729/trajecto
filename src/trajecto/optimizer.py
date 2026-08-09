import numpy as np
import json

from trajecto.problem import TrajectoryProblem

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from pymoo.parallelization.joblib import JoblibParallelization

from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from loares.experiments.utils import dict_to_csv


def merge_pareto_fronts(results):
    populations = [
        res.opt for res in results if res.opt is not None and len(res.opt) > 0
    ]

    if not populations:
        return Population.empty()

    merged = Population.merge(*populations)
    F = merged.get("F")
    rank0 = NonDominatedSorting().do(F, only_non_dominated_front=True)

    return merged[rank0]


# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=92, ref_dirs=ref_dirs)


def optimize_trajectory(
    urdf_arg,
    trajectory_function,
    time_limit,
    n_var,
    bounds,
    trajectory_extras,
    joint_limits,
    seeds,
    pymoo_algorithm=algorithm,
    n_gen=500,
    n_threads=4,
):
    # initialize the thread pool and create the runner
    runner = JoblibParallelization(n_jobs=n_threads, backend="loky")

    # define the problem by passing the starmap interface of the thread pool
    problem = TrajectoryProblem(
        urdf_arg=urdf_arg,
        trajectory_function=trajectory_function,
        time_limit=time_limit,
        n_var=n_var,
        bounds=bounds,
        trajectory_extras=trajectory_extras,
        joint_limits=joint_limits,
        elementwise_runner=runner,
    )

    results = []
    for seed in seeds:
        # execute the optimization
        res = minimize(
            problem, pymoo_algorithm, termination=("n_gen", n_gen), seed=seed
        )
        results.append(res)
        print("ExecTime:", res.exec_time)

    final_front = merge_pareto_fronts(results)

    return final_front, problem


def find_knee_point(F):
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    norm = (F - f_min) / (f_max - f_min + 1e-12)
    distances = np.sqrt(np.sum(norm**2, axis=1))
    return int(np.argmin(distances))


def save_trajectory_results(final_front, output_dir, problem):
    # Save the results to a CSV file
    F = final_front.get("F")
    X = final_front.get("X")
    G = final_front.get("G")
    results_dict = {"X": X.tolist(), "F": F.tolist(), "G": G.tolist()}
    dict_to_csv(results_dict, output_dir, "trajectory-optimization-results")

    fastest_traj_idx = np.argmin(F[:, 0])
    fastest_trajectory = problem.generate_trajectory(X[fastest_traj_idx])
    with open(f"{output_dir}/fastest-trajectory.json", "w") as f:
        json.dump(fastest_trajectory, f, indent=2, default=lambda a: a.tolist())

    efficient_traj_idx = np.argmin(F[:, 1])
    efficient_trajectory = problem.generate_trajectory(X[efficient_traj_idx])
    with open(f"{output_dir}/efficient-trajectory.json", "w") as f:
        json.dump(efficient_trajectory, f, indent=2, default=lambda a: a.tolist())

    smoothest_traj_idx = np.argmin(F[:, 2])
    smoothest_trajectory = problem.generate_trajectory(X[smoothest_traj_idx])
    with open(f"{output_dir}/smoothest-trajectory.json", "w") as f:
        json.dump(smoothest_trajectory, f, indent=2, default=lambda a: a.tolist())

    knee_idx = find_knee_point(F)
    knee_trajectory = problem.generate_trajectory(X[knee_idx])
    with open(f"{output_dir}/knee-trajectory.json", "w") as f:
        json.dump(knee_trajectory, f, indent=2, default=lambda a: a.tolist())
