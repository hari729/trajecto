import numpy as np
from pathlib import Path

# import matplotlib.pyplot as plt

from trajecto.optimizer import optimize_trajectory, save_trajectory_results
from trajecto.problem import bspline_trajecto

# from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from loares.algorithms.bxr import MO_BWR

# URDF_PATH = str(Path(__file__).parent / "ur5.urdf")
URDF_PATH = {
    "source": "package://ur_simulation_gz/urdf/ur_gz.urdf.xacro",
    "xacro_args": {"ur_type": "ur5", "name": "ur"},
}
# waypoints = np.array(
#     [
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.57, -1.0, 2.0, -1.57, 0.5, -0.8],
#         [3.14, -2.5, 1.0, 2.0, -1.2, 0.3],
#         [2.0, -3.14, 3.14, 1.0, 2.5, -2.0],
#     ]
# )
waypoints = np.array(
    [
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],  # a common UR5 "up" home pose
        [0.5, -1.3, 1.2, -1.5, -1.4, 0.0],  # pan right, elbow relaxes slightly
        [1.0, -1.0, 1.0, -1.6, -1.3, 0.3],  # continued sweep, mild wrist rotation
        [1.57, -1.57, 1.57, -1.57, -1.57, 0.0],  # mirrors start, panned 90°
    ]
)

trajectory_extras = {
    "waypoints_cont": waypoints,
    "end_v": np.zeros(6),
}

joint_limits = {
    "dq": np.full(6, 2 * np.pi),
    "ddq": np.full(6, 10.0),
    "dddq": np.full(6, 50.0),
    "tau": np.full(6, 100.0),
}


def find_knee_point(F):
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    norm = (F - f_min) / (f_max - f_min + 1e-12)
    distances = np.sqrt(np.sum(norm**2, axis=1))
    return int(np.argmin(distances))


algorithm = MO_BWR(pop_size=100)

print("Running multi-objective optimization for B-spline trajectory generation...")
final_front, problem = optimize_trajectory(
    urdf_arg=URDF_PATH,
    trajectory_function=bspline_trajecto,
    n_var=3,
    bounds=np.array([[0.5, 1.0, 1.0], [10.0, 10.0, 10.0]]),
    trajectory_extras=trajectory_extras,
    joint_limits=joint_limits,
    seeds=[1, 2],
    pymoo_algorithm=algorithm,
    n_gen=100,
    n_threads=8,
)

F = final_front.get("F")
X = final_front.get("X")

print(f"Global Pareto front: {len(F)} solutions")
print(f"Objectives [Duration, Energy, Jerk]:\n{F}")

save_trajectory_results(final_front, str(Path(__file__).parent), problem)
print(f"Results saved to {Path(__file__).parent}")

# plot = Scatter(
#     title="Global Pareto Front",
#     labels=["Duration", "Energy", "Jerk"],
#     plot_3d=True,
# )
# plot.add(F)
# plot.show()
#
# knee_idx = find_knee_point(F)
# knee_x = X[knee_idx]
# knee_f = F[knee_idx]
#
# print(f"\nKnee point (solution #{knee_idx}):")
# print(f"  x (segment durations): {knee_x}")
# print(f"  F: Duration={knee_f[0]:.3f}, Energy={knee_f[1]:.3f}, Jerk={knee_f[2]:.3f}")
#
# traj = bspline_trajecto(knee_x, trajectory_extras)
# t = traj["t"]
#
# fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#
# labels = [f"Joint {i}" for i in range(6)]
#
# axes[0].plot(t, np.rad2deg(traj["q"]))
# axes[0].set_ylabel("q [deg]")
# axes[0].set_title("Knee Point Trajectory")
# axes[0].legend(labels, fontsize=7, ncol=2)
#
# axes[1].plot(t, np.rad2deg(traj["dq"]))
# axes[1].set_ylabel("dq [deg/s]")
#
# axes[2].plot(t, np.rad2deg(traj["ddq"]))
# axes[2].set_ylabel("ddq [deg/s²]")
#
# axes[3].plot(t, np.rad2deg(traj["dddq"]))
# axes[3].set_ylabel("dddq [deg/s³]")
# axes[3].set_xlabel("Time [s]")
#
# plt.tight_layout()
# plt.show()
