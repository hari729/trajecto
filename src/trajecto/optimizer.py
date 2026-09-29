import numpy as np
import json

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

    selections = {
        "fastest": int(np.argmin(F[:, 0])),
        "efficient": int(np.argmin(F[:, 1])),
        "smoothest": int(np.argmin(F[:, 2])),
        "knee": find_knee_point(F),
    }
    for name, idx in selections.items():
        trajectory = problem.generate_trajectory(X[idx])
        with open(f"{output_dir}/{name}-trajectory.json", "w") as f:
            json.dump(trajectory, f, indent=2, default=lambda a: a.tolist())
