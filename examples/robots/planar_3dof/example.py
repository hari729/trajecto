"""Planar 3-DOF arm example — proves the pipeline is not UR5-specific.

Uses a self-contained URDF in this directory (no external ROS packages), so
the optimization runs anywhere; only run_simulation needs the container.
"""

import numpy as np
from pathlib import Path

from trajecto.config import RobotConfig, WorldConfig
from trajecto.orchestrator import Pipeline
from trajecto.samples import BSplineTrajectory

from loares.algorithms.moo.mobxr_cd import MO_BWR_CD

HERE = Path(__file__).parent

robot = RobotConfig(
    name="planar_3dof",
    urdf_source=str(HERE / "planar_3dof.urdf"),
    controllers_yaml=str(HERE / "controllers.yaml"),
)

world = WorldConfig(
    name="torque_sensor",
    sdf_path=str(HERE.parents[1] / "worlds" / "torque_sensor.sdf"),
)

waypoints = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.8, 0.6, -0.4],
        [1.6, -0.5, 0.8],
        [0.5, 1.2, -1.0],
    ]
)

# k=3 cubic splines; bconditions default; n_var defaults to the segment count
trajectory = BSplineTrajectory(waypoints=waypoints, k=3, steps=500)

# acceleration and jerk limits are not in the URDF and must be supplied
joint_limits = {
    "acceleration": np.full(3, 10.0),
    "jerk": np.full(3, 50.0),
}

pipe = Pipeline(
    robot=robot,
    world=world,
    trajectory=trajectory,
    joint_limits=joint_limits,
    time_limit=30,
    algorithm=MO_BWR_CD(pop_size=50),
    results_dir=HERE,
    seeds=[1],
    n_gen=50,
    n_threads=4,
)

if __name__ == "__main__":
    pipe.optimize()
    pipe.run_simulation(trajectory_name="knee", headless=True)
