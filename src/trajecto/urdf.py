from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def inject_ft_sensors(
    urdf_xml: str, joint_names: list[str], update_rate: int = 100
) -> str:
    root = ET.fromstring(urdf_xml)
    for name in joint_names:
        gazebo = ET.SubElement(root, "gazebo", {"reference": name})
        sensor = ET.SubElement(
            gazebo,
            "sensor",
            {
                "name": f"{name}_torque_sensor",
                "type": "force_torque",
            },
        )
        ET.SubElement(sensor, "update_rate").text = str(update_rate)
        ET.SubElement(sensor, "visualize").text = "true"
        ft = ET.SubElement(sensor, "force_torque")
        ET.SubElement(ft, "frame").text = "child"
        ET.SubElement(ft, "measure_direction").text = "parent_to_child"
        plugin = ET.SubElement(
            sensor,
            "plugin",
            {
                "filename": "gz-sim-forcetorque-system",
                "name": "gz::sim::systems::ForceTorque",
            },
        )
    return ET.tostring(root, encoding="unicode")


def load_urdf_xml(source: str, xacro_args: dict | None = None) -> str:
    """
    Resolve a URDF/xacro source into fully-expanded URDF XML.

    source may be:
      1) already-resolved XML content (e.g. a `robot_description` param value)
      2) a `package://<pkg>/relative/path` URI
      3) a plain filesystem path (.urdf or .xacro)

    xacro_args are forwarded as mappings to xacro expansion when the
    resolved file is a .xacro template (e.g. {"ur_type": "ur5"}).

    Note: raw string input (branch 1) is assumed to be fully-resolved
    URDF, not unexpanded xacro content — passing raw xacro text (rather
    than a path to a .xacro file) will not be expanded.
    """
    # 1) already XML (robot_description param, xacro output)
    if source.lstrip().startswith(("<robot", "<?xml")):
        return source

    # 2) ROS package, e.g. package://ur_description/urdf/ur5/ur5.urdf.xacro
    if source.startswith("package://"):
        pkg, _, rel = source[len("package://") :].partition("/")
        from ament_index_python.packages import get_package_share_directory  # lazy

        path = Path(get_package_share_directory(pkg)) / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"Resolved {source!r} to {path}, but it doesn't exist"
            )

    # 3) plain file path
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {source}")

    # 4) xacro template -> expand to URDF XML
    if path.suffix == ".xacro":
        import xacro  # lazy

        return xacro.process_file(str(path), mappings=xacro_args or {}).toxml()

    return path.read_text()


def set_initial_joint_positions(
    urdf_xml: str, joint_names: list[str], start_q: list[float]
) -> str:
    root = ET.fromstring(urdf_xml)
    if root.find(".//ros2_control") is None:
        raise ValueError(
            "URDF has no <ros2_control> block, so initial joint positions "
            "cannot be injected. Add a ros2_control system with a position "
            "state_interface per joint, or extend the launch to pass initial "
            "positions to the spawner."
        )
    for jn, q0 in zip(joint_names, start_q):
        joint_el = root.find(f".//ros2_control//joint[@name='{jn}']")
        if joint_el is None:
            raise ValueError(f"joint '{jn}' not found in <ros2_control> block")
        pos_iface = joint_el.find("state_interface[@name='position']")
        if pos_iface is None:
            raise ValueError(f"joint '{jn}' has no position state_interface")
        param = pos_iface.find("param[@name='initial_value']")
        if param is None:
            param = ET.SubElement(pos_iface, "param", {"name": "initial_value"})
        param.text = str(q0)
    return ET.tostring(root, encoding="unicode")


def _movable_joints(root: ET.Element) -> dict[str, ET.Element]:
    """Map movable-joint name -> its <joint> element, in URDF declaration order."""
    return {
        j.get("name"): j
        for j in root.findall("joint")
        if j.get("type") in ("revolute", "continuous", "prismatic")
    }


def parse_joint_limits(urdf_xml: str, joint_names: list[str]) -> dict:
    """Derive per-joint position/velocity/effort limits from URDF <limit>.

    Returns {"velocity": array, "torque": array, "position": (2, n) array}
    in the given joint_names order. The position array has row 0 as the lower
    limit and row 1 as the upper limit.

    Raises if a requested joint is missing or has a non-positive
    (unspecified) velocity/effort limit, or a missing/unspecified position
    bound, since a zero or missing limit would make every trajectory
    infeasible.
    """
    joints = _movable_joints(ET.fromstring(urdf_xml))
    velocity, torque, position_lower, position_upper = [], [], [], []
    for jn in joint_names:
        el = joints.get(jn)
        lim = el.find("limit") if el is not None else None
        v = float(lim.get("velocity")) if lim is not None else 0.0
        e = float(lim.get("effort")) if lim is not None else 0.0
        lower = lim.get("lower") if lim is not None else None
        upper = lim.get("upper") if lim is not None else None
        if v <= 0:
            raise ValueError(
                f"joint '{jn}' has no positive <limit velocity=...> in the URDF; "
                "supply a velocity limit via RobotConfig.joint_limits"
            )
        if e <= 0:
            raise ValueError(
                f"joint '{jn}' has no positive <limit effort=...> in the URDF; "
                "supply a torque limit via RobotConfig.joint_limits"
            )
        if lower is None or upper is None:
            raise ValueError(
                f"joint '{jn}' has no <limit lower=... upper=...> in the URDF; "
                "supply a position limit via RobotConfig.joint_limits"
            )
        velocity.append(v)
        torque.append(e)
        position_lower.append(float(lower))
        position_upper.append(float(upper))
    return {
        "velocity": velocity,
        "torque": torque,
        "position": np.array([position_lower, position_upper], dtype=float),
    }


def parse_joint_axes(urdf_xml: str, joint_names: list[str]) -> dict[str, list[float]]:
    """Return each joint's rotation axis as [x, y, z] (URDF default is 1 0 0)."""
    joints = _movable_joints(ET.fromstring(urdf_xml))
    axes = {}
    for jn in joint_names:
        el = joints.get(jn)
        axis = el.find("axis") if el is not None else None
        xyz = axis.get("xyz").split() if axis is not None else ["1", "0", "0"]
        axes[jn] = [float(c) for c in xyz]
    return axes
