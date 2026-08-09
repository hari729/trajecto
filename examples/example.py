import numpy as np
from pathlib import Path

from trajecto.orchestrator import Pipeline
from trajecto.problem import bspline_trajecto

from loares.algorithms.bxr import MO_BWR

from ament_index_python.packages import get_package_share_directory

controllers_yaml_path = str(
    Path(get_package_share_directory("ur_simulation_gz"))
    / "config"
    / "ur_controllers.yaml"
)

URDF_PATH = {
    "source": "package://ur_simulation_gz/urdf/ur_gz.urdf.xacro",
    "xacro_args": {
        "ur_type": "ur5",
        "name": "ur",
        "simulation_controllers": controllers_yaml_path,
    },
}

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
pipe = Pipeline(
    robot_name="ur",
    urdf_arg=URDF_PATH,
    waypoints=waypoints,
    trajectory_generator=bspline_trajecto,
    joint_limits=joint_limits,
    trajectory_extras=trajectory_extras,
    n_var=3,
    var_bounds=np.array([[0.5, 1.0, 1.0], [10.0, 10.0, 10.0]]),
    algorithm=MO_BWR(pop_size=100),
    results_dir=Path(__file__).parent,
    seeds=[1, 2],
    n_gen=100,
    n_threads=8,
)

# pipe.optimize()
# pipe.launch_simulation(
#     controllers_yaml_path=controllers_yaml_path,
#     world_name="torque_sensor",
#     world_file=str(Path(__file__).parent / "torque_sensor.sdf"),
# )
# pipe.simulate_trajectory("knee", show=True)
# pipe.shutdown_simulation()
# pipe.optimize()

trajectory_name = "fastest"

pipe.run_simulation(
    controllers_yaml_path=controllers_yaml_path,
    world_name="torque_sensor",
    world_file=str(Path(__file__).parent / "torque_sensor.sdf"),
    trajectory_name=trajectory_name,
)

from trajecto.plots import plot_rnea_on_measured

plot_rnea_on_measured(
    str(pipe.results_dir / f"{trajectory_name}-trajectory.json"),
    str(pipe.simul_results_dir / f"{trajectory_name}-joint-states.json"),
    str(pipe.simul_results_dir / f"{trajectory_name}-ft-sensor.json"),
    rmodel=pipe.problem.robotmodel,
)
