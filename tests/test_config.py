from pathlib import Path

import numpy as np
import pytest

from trajecto.config import RobotConfig, WorldConfig


class TestRobotConfig:
    def test_minimal_config_uses_defaults(self, tmp_path):
        yaml = tmp_path / "controllers.yaml"
        yaml.write_text("controller_manager: {}")
        cfg = RobotConfig(
            name="ur",
            urdf_source="package://ur_simulation_gz/urdf/ur_gz.urdf.xacro",
            controllers_yaml=str(yaml),
        )
        assert cfg.controller_name == "joint_trajectory_controller"
        assert cfg.xacro_args is None
        assert cfg.joint_limits is None
        assert cfg.home_pose is None

    def test_empty_name_rejected(self, tmp_path):
        yaml = tmp_path / "controllers.yaml"
        yaml.write_text("x")
        with pytest.raises(ValueError, match="name"):
            RobotConfig(name="", urdf_source="package://x", controllers_yaml=str(yaml))

    def test_missing_local_urdf_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "controllers.yaml").write_text("x")
        with pytest.raises(FileNotFoundError, match="urdf_source"):
            RobotConfig(
                name="r",
                urdf_source="missing.urdf",
                controllers_yaml="controllers.yaml",
            )

    def test_missing_local_controllers_yaml_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="controllers_yaml"):
            RobotConfig(
                name="r",
                urdf_source="package://pkg/robot.urdf",
                controllers_yaml="missing.yaml",
            )

    def test_absolute_paths_not_existence_checked(self):
        # absolute ROS paths may only exist inside the container; no error
        cfg = RobotConfig(
            name="ur",
            urdf_source="package://ur_simulation_gz/urdf/ur_gz.urdf.xacro",
            controllers_yaml="/opt/ros/jazzy/share/ur_simulation_gz/config/ur_controllers.yaml",
        )
        assert cfg.name == "ur"


class TestWorldConfig:
    def test_valid_world(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sdf = tmp_path / "world.sdf"
        sdf.write_text("<sdf version='1.6'><world name='w'/></sdf>")
        cfg = WorldConfig(name="w", sdf_path="world.sdf")
        assert cfg.name == "w"

    def test_missing_sdf_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="sdf_path"):
            WorldConfig(name="w", sdf_path="nope.sdf")

    def test_empty_name_rejected(self, tmp_path):
        sdf = tmp_path / "world.sdf"
        sdf.write_text("<sdf/>")
        with pytest.raises(ValueError, match="name"):
            WorldConfig(name="", sdf_path=str(sdf))
