from launch import LaunchService
from pathlib import Path
from trajecto.problem import RobotModel
from trajecto.launch_sim import build_robot_launch

from ament_index_python.packages import get_package_share_directory

controllers_yaml_path = str(
    Path(get_package_share_directory("ur_simulation_gz"))
    / "config"
    / "ur_controllers.yaml"
)

controller_name = "joint_trajectory_controller"
URDF_PATH = {
    "source": "package://ur_simulation_gz/urdf/ur_gz.urdf.xacro",
    "xacro_args": {
        "ur_type": "ur5",
        "name": "ur",
        "simulation_controllers": controllers_yaml_path,
    },
}

rmodel = RobotModel(**URDF_PATH)

ld = build_robot_launch(
    robot_model=rmodel,
    controllers_yaml_path=controllers_yaml_path,
    controller_name=controller_name,
    world_name="torque_sensor",
    world_file=str(Path(__file__).parent / "torque_sensor.sdf"),
)
ls = LaunchService()
ls.include_launch_description(ld)
ls.run()
