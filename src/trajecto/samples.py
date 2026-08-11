import numpy as np


def bspline_trajectory(x, num_joints, k, waypoints, bconditions, steps=500):
    from scipy.interpolate import make_interp_spline

    t = np.concatenate(([0.0], np.cumsum(x)))
    t_fine = np.linspace(t.min(), t.max(), steps)
    q_out, dq_out, ddq_out, dddq_out = [], [], [], []

    for i in range(num_joints):
        spline = make_interp_spline(t, waypoints[:, i], k=k, bc_type=bconditions[i])

        q_out.append(spline(t_fine))
        dq_out.append(spline(t_fine, nu=1))
        ddq_out.append(spline(t_fine, nu=2))
        dddq_out.append(spline(t_fine, nu=3))

    trajectory = {}
    trajectory["time"] = t_fine
    trajectory["position"] = np.array(q_out).T
    trajectory["velocity"] = np.array(dq_out).T
    trajectory["acceleration"] = np.array(ddq_out).T
    trajectory["jerk"] = np.array(dddq_out).T

    return trajectory


def bspline_trajecto(x, trajectory_extras):

    end_derivs = [
        ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
        (
            [(1, 0.0), (2, 0.0), (3, 0.0)],
            [(1, trajectory_extras["end_v"][1]), (2, 0.0)],
        ),
        (
            [(1, 0.0), (2, 0.0), (3, 0.0)],
            [(1, trajectory_extras["end_v"][2]), (2, 0.0)],
        ),
        (
            [(1, 0.0), (2, 0.0), (3, 0.0)],
            [(1, trajectory_extras["end_v"][3]), (2, 0.0)],
        ),
        ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
        ([(1, 0.0), (2, 0.0), (3, 0.0)], [(1, 0.0), (2, 0.0)]),
    ]

    btrajectory = bspline_trajectory(
        np.concatenate(([0.0], np.cumsum(x))),
        6,
        6,
        trajectory_extras["waypoints_cont"],
        end_derivs,
    )

    return btrajectory
