from pathlib import Path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory

import yaml
import tempfile


def write_bridge_config(
    world_name: str, robot_name: str, joint_names: list[str]
) -> str:
    entries = [
        {
            "ros_topic_name": "/clock",
            "gz_topic_name": "/clock",
            "ros_type_name": "rosgraph_msgs/msg/Clock",
            "gz_type_name": "gz.msgs.Clock",
            "direction": "GZ_TO_ROS",
        }
    ]
    for jn in joint_names:
        entries.append(
            {
                "ros_topic_name": f"/{robot_name}/{jn}/force_torque",
                "gz_topic_name": f"/world/{world_name}/model/{robot_name}/joint/{jn}/sensor/{jn}_torque_sensor/forcetorque",
                "ros_type_name": "geometry_msgs/msg/Wrench",
                "gz_type_name": "gz.msgs.Wrench",
                "direction": "GZ_TO_ROS",
            }
        )

    fd, path = tempfile.mkstemp(suffix="_bridge.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(entries, f)
    return path


def build_robot_launch(
    robot_model,
    controllers_yaml_path: str,
    controller_name: str,
    robot_name: str = "robot",
    world_name: str = "empty",
    world_file: str = "empty.sdf",
) -> LaunchDescription:
    """
    urdf_xml: fully-resolved URDF/SDF-ready XML string (already through
        load_urdf_xml + any injection steps like FT sensors).
    controllers_yaml_path: path to a controller_manager YAML — user-supplied
        or generated from the same joint-name list used to build urdf_xml.
    controller_name: the controller to spawn from that YAML, e.g.
        'joint_trajectory_controller' — varies per controller type/user
        choice, so it's a parameter, not a constant.
    """
    urdf_xml = robot_model.urdf_xml

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, {"robot_description": urdf_xml}],
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", urdf_xml, "-name", robot_name, "-allow_renaming", "true"],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[controller_name, "-c", "/controller_manager"],
        parameters=[ParameterFile(controllers_yaml_path, allow_substs=True)],
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                Path(get_package_share_directory("ros_gz_sim"))
                / "launch"
                / "gz_sim.launch.py"
            )
        ),
        launch_arguments={"gz_args": f"-r -v 4 {world_file}"}.items(),
    )

    bridge_config_path = write_bridge_config(
        world_name, robot_name, robot_model.joint_names
    )
    gz_sim_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["--ros-args", "-p", f"config_file:={bridge_config_path}"],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            gz_spawn_entity,
            joint_state_broadcaster_spawner,
            controller_spawner,
            gz_sim,
            gz_sim_bridge,
        ]
    )
