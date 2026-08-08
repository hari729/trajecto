from pathlib import Path
import xml.etree.ElementTree as ET


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
