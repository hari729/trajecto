def plot_trajectory_comparison(
    trajectory_path: str,
    joint_states_path: str,
    ft_readings_path: str,
    output_path: str = "comparison.png",
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    with open(trajectory_path) as f:
        kt = json.load(f)
    with open(joint_states_path) as f:
        js = json.load(f)

    joint_names = kt["joint_names"]
    n_joints = len(joint_names)

    t_cmd = np.array(kt["t"])
    q_cmd = np.array(kt["q"])
    dq_cmd = np.array(kt["dq"])
    tau_cmd = np.array(kt["tau"])

    t_meas = np.array([s["time"] for s in js])
    t_meas = t_meas - t_meas[0]

    # each recorded sample carries its own name order — align to
    # trajectory_path's joint_names per sample rather than assuming
    # a fixed order across the whole recording
    q_meas = np.array(
        [[s["position"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    dq_meas = np.array(
        [[s["velocity"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    tau_meas = np.array(
        [[s["effort"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )

    signals = [
        ("position [rad]", q_cmd, q_meas, "commanded"),
        ("velocity [rad/s]", dq_cmd, dq_meas, "commanded"),
        ("effort [N·m]", tau_cmd, tau_meas, "RNEA (theoretical)"),
    ]

    fig, axes = plt.subplots(n_joints, 3, figsize=(15, 3 * n_joints))
    for row, jname in enumerate(joint_names):
        for col, (ylabel, cmd, meas, cmd_label) in enumerate(signals):
            ax = axes[row, col]
            ax.plot(
                t_cmd, cmd[:, row], label=cmd_label, color="tab:blue", linewidth=1.5
            )
            ax.plot(
                t_meas,
                meas[:, row],
                label="measured",
                color="tab:orange",
                linewidth=1.2,
            )
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(ylabel.split(" [")[0].capitalize())
            if row == n_joints - 1:
                ax.set_xlabel("time [s]")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        axes[row, 0].annotate(
            jname,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="center",
        )

    plt.tight_layout()
    plt.show()
    # # plt.savefig(output_path, dpi=130)
    # # plt.close(fig)
    # return output_path


def plot_rnea_on_measured(
    trajectory_path: str,
    joint_states_path: str,
    ft_sensor_path: str,
    rmodel,
):
    """
    Recompute RNEA using the *measured* q, dq (with ddq estimated via
    finite differences) instead of the commanded trajectory's q, dq, ddq.
    Plots four curves per joint: RNEA-on-commanded, RNEA-on-measured,
    /joint_states effort (simulated actuator's internal commanded
    effort — not a physical torque measurement), and the true FT
    sensor reading — isolating whether torque mismatch comes from the
    controller not reproducing the planned motion, versus an actual
    dynamics/measurement discrepancy.
    """
    import json
    import numpy as np
    import pinocchio as pin
    import matplotlib.pyplot as plt

    with open(trajectory_path) as f:
        kt = json.load(f)
    with open(joint_states_path) as f:
        js = json.load(f)
    with open(ft_sensor_path) as f:
        ft = json.load(f)

    joint_names = kt["joint_names"]
    n_joints = len(joint_names)
    t_cmd = np.array(kt["t"])
    tau_cmd = np.array(kt["tau"])

    t_meas_abs = np.array([s["time"] for s in js])
    t0 = t_meas_abs[0]
    t_meas = t_meas_abs - t0

    q_meas = np.array(
        [[s["position"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    dq_meas = np.array(
        [[s["velocity"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )
    effort_meas = np.array(
        [[s["effort"][s["name"].index(jn)] for jn in joint_names] for s in js]
    )

    ddq_meas = np.gradient(dq_meas, t_meas, axis=0)

    model = pin.buildModelFromXML(rmodel.urdf_xml)
    data = model.createData()
    tau_rnea_on_measured = np.array(
        [
            pin.rnea(model, data, q_meas[i], dq_meas[i], ddq_meas[i])
            for i in range(len(t_meas))
        ]
    )

    # ft_sensor.json: dict keyed by joint name, each a list of
    # {time, torque} samples with its own irregular timestamps
    ft_data = {}
    for jn in joint_names:
        samples = ft[jn]
        t_ft_abs = np.array([s["time"] for s in samples])
        t_ft = t_ft_abs - t0
        tau_ft = np.array([s["torque"] for s in samples])
        ft_data[jn] = (t_ft, tau_ft)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    axes = axes.flatten()
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
            effort_meas[:, row],
            label="/joint_states effort (actuator, not FT)",
            color="tab:gray",
            linewidth=1.0,
            alpha=0.7,
        )
        t_ft, tau_ft = ft_data[jname]
        ax.plot(
            t_ft,
            tau_ft,
            label="FT sensor (measured)",
            color="tab:orange",
            linewidth=1.0,
        )
        ax.set_title(jname)
        ax.set_ylabel("torque [N·m]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[-2].set_xlabel("time [s]")
    axes[-1].set_xlabel("time [s]")

    plt.tight_layout()
    plt.show()
