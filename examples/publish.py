from pathlib import Path
from rclpy.executors import MultiThreadedExecutor
from trajecto.nodes import publish_trajectory, record_joint_states_new
import rclpy

rclpy.init()

node = publish_trajectory(str(Path(__file__).parent / "knee_trajectory.json"))
recorder = record_joint_states_new(
    [
        str(Path(__file__).parent / "joint_states.json"),
        str(Path(__file__).parent / "ft_sensor.json"),
    ],
    joint_names=[
        "elbow_joint",
        "shoulder_lift_joint",
        "shoulder_pan_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
    robot_name="robot",
)

executor = MultiThreadedExecutor()
executor.add_node(node)
executor.add_node(recorder)

result = node.send_trajectory(executor)
recorder.save_readings()

node.destroy_node()
recorder.destroy_node()
rclpy.shutdown()
