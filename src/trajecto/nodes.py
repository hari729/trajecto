import enum
from os.path import join
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)  # Message types for joint trajectories
from builtin_interfaces.msg import Duration  # Message type for time durations
from sensor_msgs.msg import JointState
import json

import rclpy  # ROS 2 Python client library
from rclpy.node import Node  # Base class for ROS 2 nodes
import time  # For sleep/delay

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Wrench
from rclpy.parameter import Parameter


def generate_trajectory_msg(trajectory_file_path):
    with open(trajectory_file_path, "r") as file:
        trajectory = json.load(file)

    trajectory_msg = JointTrajectory()
    if trajectory["joint_names"] is not None:
        trajectory_msg.joint_names = trajectory["joint_names"]
        # print(trajectory_msg.joint_names)
    for i in range(len(trajectory["time"])):
        tmsg_point = JointTrajectoryPoint()
        tmsg_point.positions = trajectory["position"][i]
        tmsg_point.velocities = trajectory["velocity"][i]
        tmsg_point.accelerations = trajectory["acceleration"][i]
        tmsg_point.time_from_start = Duration(
            sec=int(trajectory["time"][i]),
            nanosec=int((trajectory["time"][i] % 1) * 1e9),
        )
        trajectory_msg.points.append(tmsg_point)

    return trajectory_msg


class publish_trajectory(Node):
    def __init__(self, trajectory_file_path):
        super().__init__("publish_trajectory")
        self.get_logger().info("publish_trajectory node has been started")
        self.trajectory_file_path = trajectory_file_path
        self.trajectory = generate_trajectory_msg(trajectory_file_path)
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            "joint_trajectory_controller/follow_joint_trajectory",
        )

    def send_trajectory(self, executor):
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available after waiting")
            return None

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self.trajectory
        self.get_logger().info("Sending trajectory goal...")

        send_future = self._client.send_goal_async(goal)
        executor.spin_until_future_complete(send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return None

        result_future = goal_handle.get_result_async()
        executor.spin_until_future_complete(result_future)
        return result_future.result()


class record_joint_states(Node):
    def __init__(self, output_file_path, joint_names, robot_name="robot"):
        super().__init__(
            "record_joint_states",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self.output_file_path = output_file_path
        self.joint_names = joint_names
        self.joint_states = {
            jn: {"position": [], "velocity": [], "torque": []} for jn in joint_names
        }
        self.joint_states["time"] = []
        self.ft_readings = {jn: {"time": [], "torque": []} for jn in joint_names}

        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        for jn in joint_names:
            self.create_subscription(
                Wrench,
                f"/{robot_name}/{jn}/force_torque",
                lambda msg, jn=jn: self.ft_callback(msg, jn),
                10,
            )

    def joint_state_callback(self, msg):
        self.joint_states["time"].append(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        for idx, name in enumerate(msg.name):
            self.joint_states[name]["position"].append(msg.position[idx])
            self.joint_states[name]["velocity"].append(msg.velocity[idx])
            self.joint_states[name]["torque"].append(msg.effort[idx])

    def ft_callback(self, msg, joint_name):
        self.ft_readings[joint_name]["time"].append(
            self.get_clock().now().nanoseconds * 1e-9
        )
        self.ft_readings[joint_name]["torque"].append(
            msg.torque.z
        )  # confirm this is the axis aligned with the joint's rotation axis

    def save_readings(self):
        with open(self.output_file_path[0], "w") as file:
            json.dump(
                self.joint_states,
                file,
                indent=2,
                default=lambda a: a.tolist(),
            )

        with open(self.output_file_path[1], "w") as file:
            json.dump(self.ft_readings, file, indent=2, default=lambda a: a.tolist())
