def plot_trajectory_comparison(
    trajectory_path: str,
    joint_states_path: str,
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
