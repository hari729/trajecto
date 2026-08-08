def plot_rnea_on_measured(
    trajectory_path: str,
    joint_states_path: str,
    rmodel,
):
    """
    Recompute RNEA using the *measured* q, dq (with ddq estimated via
    finite differences) instead of the commanded trajectory's q, dq, ddq.
    Plots three curves per joint: RNEA-on-commanded, RNEA-on-measured,
    and the FT-sensor-measured effort — isolating whether torque
    mismatch comes from the controller not reproducing the planned
    motion, versus an actual dynamics/measurement discrepancy.
    """
    import json
    import numpy as np
    import pinocchio as pin

    # import matplotlib
    #
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(trajectory_path) as f:
        kt = json.load(f)
    with open(joint_states_path) as f:
        js = json.load(f)

    joint_names = kt["joint_names"]
    n_joints = len(joint_names)

    t_cmd = np.array(kt["t"])
    tau_cmd = np.array(kt["tau"])

    t_meas = np.array([s["time"] for s in js])
    t_meas = t_meas - t_meas[0]

    q_meas = np.array(
        [[s["position"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    dq_meas = np.array(
        [[s["velocity"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    tau_meas = np.array(
        [[s["effort"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )

    # estimate ddq from measured velocity via finite differences —
    # handles non-uniform timestamps correctly, unlike a fixed dt
    ddq_meas = np.gradient(dq_meas, t_meas, axis=0)

    model = pin.buildModelFromXML(rmodel.urdf_xml)
    data = model.createData()

    tau_rnea_on_measured = np.array(
        [
            pin.rnea(model, data, q_meas[i], dq_meas[i], ddq_meas[i])
            for i in range(len(t_meas))
        ]
    )

    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    axes = axes.flatten()
    if n_joints == 1:
        axes = [axes]
    for row, jname in enumerate(joint_names):
        ax = axes[row]
        ax.plot(
            t_cmd,
            tau_cmd[:, row],
            label="RNEA (commanded)",
            color="tab:blue",
            linewidth=1.5,
        )
        ax.plot(
            t_meas,
            tau_rnea_on_measured[:, row],
            label="RNEA (measured q, dq)",
            color="tab:green",
            linewidth=1.5,
        )
        ax.plot(
            t_meas,
            tau_meas[:, row],
            label="FT sensor (measured)",
            color="tab:orange",
            linewidth=1.2,
        )
        ax.set_title(jname)
        ax.set_ylabel("torque [N·m]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    from trajecto.problem import RobotModel

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
    print(rmodel.model.gravity)

    # plot_rnea_on_measured(
    #     str(Path(__file__).parent / "knee_trajectory.json"),
    #     str(Path(__file__).parent / "joint_states.json"),
    #     rmodel=rmodel,
    # )
