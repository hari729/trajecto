"""Declarative robot/world configuration for a Pipeline run.

Swapping robots means constructing a different RobotConfig, not editing the
library. See examples/robots/ for concrete instances.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RobotConfig:
    """Everything robot-specific needed to optimize and simulate a trajectory.

    name:             robot/model name used for the Gazebo spawn, topic
                      prefixes and the results subdirectory.
    urdf_source:      resolved URDF XML, a filesystem path (.urdf/.xacro), or
                      a `package://` URI — anything `load_urdf_xml` accepts.
    xacro_args:       forwarded to xacro when `urdf_source` is a .xacro.
    controllers_yaml: controller_manager YAML defining the controller(s).
    controller_name:  controller to spawn (must exist in controllers_yaml).
    joint_limits:     optional overrides for the limits derived from the URDF;
                      keys are any of "velocity", "acceleration", "jerk",
                      "torque", or "position" with per-joint array values
                      (URDF order). "position" is a (2, n_joints) array where
                      row 0 is the lower bound and row 1 is the upper bound.
    home_pose:        optional joint-space home configuration (URDF order).
    """

    name: str
    urdf_source: str
    controllers_yaml: str
    controller_name: str = "joint_trajectory_controller"
    xacro_args: dict[str, Any] | None = None
    joint_limits: dict[str, Any] | None = None
    home_pose: list[float] | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("RobotConfig.name must be non-empty")
        if not self.urdf_source:
            raise ValueError("RobotConfig.urdf_source must be non-empty")
        if not self.controllers_yaml:
            raise ValueError("RobotConfig.controllers_yaml must be non-empty")
        if not self.controller_name:
            raise ValueError("RobotConfig.controller_name must be non-empty")
        # controllers_yaml may point at a ROS package file that only exists
        # inside the container (an absolute path); only check existence for
        # relative local paths.
        if not Path(self.urdf_source).is_absolute() and not (
            self.urdf_source.startswith(("package://", "<"))
            or Path(self.urdf_source).is_file()
        ):
            raise FileNotFoundError(
                f"RobotConfig.urdf_source not found: {self.urdf_source!r}"
            )
        # controllers_yaml may point at a ROS package file that only exists
        # inside the container (an absolute path); only check existence for
        # relative local paths.
        if not Path(self.controllers_yaml).is_absolute() and not Path(
            self.controllers_yaml
        ).is_file():
            raise FileNotFoundError(
                f"RobotConfig.controllers_yaml not found: {self.controllers_yaml!r}"
            )


@dataclass
class WorldConfig:
    """A Gazebo world to simulate in.

    name:     world name (must match the <world name="..."> in the SDF); used
              for the FT-sensor bridge topics and the sim results subdirectory.
    sdf_path: path to the world SDF file.
    """

    name: str
    sdf_path: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("WorldConfig.name must be non-empty")
        if not Path(self.sdf_path).is_file():
            raise FileNotFoundError(
                f"WorldConfig.sdf_path not found: {self.sdf_path!r}"
            )


@dataclass
class RobotWorldPair:  # pragma: no cover - convenience container
    robot: RobotConfig
    world: WorldConfig
    extra: dict[str, Any] = field(default_factory=dict)
