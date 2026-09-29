import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from trajecto.urdf import (
    inject_ft_sensors,
    load_urdf_xml,
    parse_joint_axes,
    parse_joint_limits,
    set_initial_joint_positions,
)

URDF_PATH = Path(__file__).parent / "ur5.urdf"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class TestLoadUrdfXml:
    def test_raw_xml_is_returned_unchanged(self):
        xml = '<robot name="dummy"></robot>'
        assert load_urdf_xml(xml) == xml

    def test_file_path_is_loaded(self):
        xml = load_urdf_xml(str(URDF_PATH))
        assert "<robot" in xml
        assert "shoulder_pan_joint" in xml

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_urdf_xml("/nonexistent/robot.urdf")


class TestInjectFtSensors:
    def test_one_sensor_per_joint_with_matching_names(self):
        xml = inject_ft_sensors(load_urdf_xml(str(URDF_PATH)), JOINT_NAMES)
        root = ET.fromstring(xml)

        gazebo_els = root.findall("gazebo")
        assert {g.get("reference") for g in gazebo_els} == set(JOINT_NAMES)
        assert len(gazebo_els) == len(JOINT_NAMES)

        for jn in JOINT_NAMES:
            gazebo = next(g for g in gazebo_els if g.get("reference") == jn)
            sensor = gazebo.find("sensor")
            assert sensor.get("name") == f"{jn}_torque_sensor"
            assert sensor.get("type") == "force_torque"
            assert sensor.findtext("update_rate") == "100"

    def test_original_joints_preserved(self):
        xml = inject_ft_sensors(load_urdf_xml(str(URDF_PATH)), JOINT_NAMES[:1])
        root = ET.fromstring(xml)
        joint_names = {j.get("name") for j in root.findall("joint")}
        assert set(JOINT_NAMES) <= joint_names


class TestSetInitialJointPositions:
    def test_sets_initial_value_params(self):
        xml = load_urdf_xml(str(URDF_PATH))
        start_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        out = set_initial_joint_positions(xml, JOINT_NAMES, start_q)
        root = ET.fromstring(out)

        for jn, q0 in zip(JOINT_NAMES, start_q):
            joint_el = root.find(f".//ros2_control//joint[@name='{jn}']")
            pos_iface = joint_el.find("state_interface[@name='position']")
            param = pos_iface.find("param[@name='initial_value']")
            assert param is not None
            assert float(param.text) == pytest.approx(q0)
            # exactly one initial_value param (the original must be replaced,
            # not shadowed by an appended duplicate)
            assert len(pos_iface.findall("param[@name='initial_value']")) == 1

    def test_unknown_joint_raises(self):
        xml = load_urdf_xml(str(URDF_PATH))
        with pytest.raises(ValueError, match="no_such_joint"):
            set_initial_joint_positions(xml, ["no_such_joint"], [0.0])


class TestParseJointLimits:
    def test_derives_velocity_and_effort(self):
        xml = load_urdf_xml(str(URDF_PATH))
        limits = parse_joint_limits(xml, JOINT_NAMES)
        np.testing.assert_allclose(limits["velocity"], [np.pi] * 6)
        np.testing.assert_allclose(limits["torque"], [150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    def test_missing_or_zero_limit_raises(self):
        xml = (
            '<robot name="r">'
            '<link name="a"/><link name="b"/>'
            '<joint name="j1" type="revolute"><parent link="a"/><child link="b"/>'
            '<limit lower="-1" upper="1" effort="0" velocity="0"/></joint>'
            "</robot>"
        )
        with pytest.raises(ValueError, match="velocity"):
            parse_joint_limits(xml, ["j1"])

    def test_unknown_joint_raises(self):
        xml = load_urdf_xml(str(URDF_PATH))
        with pytest.raises(ValueError):
            parse_joint_limits(xml, ["no_such_joint"])


class TestParseJointAxes:
    def test_ur5_all_z_axis(self):
        xml = load_urdf_xml(str(URDF_PATH))
        axes = parse_joint_axes(xml, JOINT_NAMES)
        for jn in JOINT_NAMES:
            assert axes[jn] == [0.0, 0.0, 1.0]

    def test_default_axis_is_x(self):
        xml = (
            '<robot name="r">'
            '<link name="a"/><link name="b"/>'
            '<joint name="j1" type="revolute"><parent link="a"/><child link="b"/>'
            '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            "</robot>"
        )
        assert parse_joint_axes(xml, ["j1"])["j1"] == [1.0, 0.0, 0.0]

    def test_custom_axis(self):
        xml = (
            '<robot name="r">'
            '<link name="a"/><link name="b"/>'
            '<joint name="j1" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 1 0"/>'
            '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            "</robot>"
        )
        assert parse_joint_axes(xml, ["j1"])["j1"] == [0.0, 1.0, 0.0]
