"""Trajectory models.

`Trajectory` is the extension point — like pymoo's `Problem`, you subclass it
and implement `_generate(x)`. `Pipeline`/`TrajectoryProblem` take a Trajectory
instance, which carries its own optimization variables (`n_var`) and `bounds`.

`BSplineTrajectory` is the shipped reference implementation (cubic-and-higher
B-splines through joint-space waypoints, optimized segment durations).
"""

from abc import ABC, abstractmethod

import numpy as np

REQUIRED_KEYS = ("time", "position", "velocity", "acceleration", "jerk")


class Trajectory(ABC):
    """A parametric trajectory model.

    Subclasses implement `_generate(x)` returning a dict with the keys
    `time`, `position`, `velocity`, `acceleration`, `jerk`. `n_var` is the
    dimension of `x` and `bounds` its (2, n_var) lower/upper bounds, consumed
    by the optimizer. Subclasses must be picklable (module-level) because the
    problem is evaluated in joblib worker processes.
    """

    def __init__(self, n_var, bounds):
        self.n_var = int(n_var)
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (2, self.n_var):
            raise ValueError(
                f"bounds must have shape (2, {self.n_var}), got {self.bounds.shape}"
            )

    @abstractmethod
    def _generate(self, x):
        """Return the trajectory dict for decision vector `x`."""
        raise NotImplementedError

    def __call__(self, x):
        traj = self._generate(np.asarray(x, dtype=float))
        self._validate(traj)
        return traj

    def _validate(self, traj):
        if not isinstance(traj, dict):
            raise TypeError(
                f"{type(self).__name__}._generate must return a dict, got "
                f"{type(traj).__name__}"
            )
        missing = [k for k in REQUIRED_KEYS if k not in traj]
        if missing:
            raise ValueError(
                f"{type(self).__name__} trajectory is missing keys: {missing} "
                f"(need {list(REQUIRED_KEYS)})"
            )
        for key in REQUIRED_KEYS:
            arr = np.asarray(traj[key])
            traj[key] = arr
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"trajectory[{key!r}] contains non-finite values")
        t = traj["time"]
        if t.ndim != 1:
            raise ValueError(f"trajectory['time'] must be 1-D, got shape {t.shape}")
        n_steps = t.shape[0]
        if n_steps < 2:
            raise ValueError("trajectory needs at least 2 time steps")
        if not np.all(np.diff(t) > 0):
            raise ValueError("trajectory['time'] must be strictly increasing")
        for key in ("position", "velocity", "acceleration", "jerk"):
            arr = traj[key]
            if arr.ndim != 2 or arr.shape[0] != n_steps:
                raise ValueError(
                    f"trajectory[{key!r}] must have shape (n_steps, n_joints); "
                    f"got {arr.shape} for n_steps={n_steps}"
                )


class BSplineTrajectory(Trajectory):
    """Joint-space B-spline through waypoints; x scales the segment durations.

    waypoints: (n_waypoints, n_joints) in URDF joint order. n_var defaults to
    the number of segments (n_waypoints - 1); bounds default to a per-segment
    duration in [default_dt_min, default_dt_max]. bconditions defaults to
    clamped (zero 1st–3rd derivatives at the start, zero 1st–2nd at the end).
    """

    def __init__(
        self,
        waypoints,
        k=3,
        steps=500,
        bconditions=None,
        n_var=None,
        bounds=None,
        default_dt_min=0.5,
        default_dt_max=20.0,
    ):
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2:
            raise ValueError(
                f"waypoints must be (n_waypoints, n_joints), got {waypoints.shape}"
            )
        self.waypoints = waypoints
        self.n_joints = waypoints.shape[1]
        self.k = int(k)
        self.steps = int(steps)
        n_segments = waypoints.shape[0] - 1
        if n_var is None:
            n_var = n_segments
        if bounds is None:
            bounds = np.array(
                [np.full(n_var, default_dt_min), np.full(n_var, default_dt_max)]
            )
        if bconditions is None:
            # clamped start, smooth end; order capped at the spline degree k.
            # scipy requires nt - n == len(left) + len(right); if the default
            # is inconsistent for these waypoints, fall back to scipy's auto.
            left = [(i + 1, 0.0) for i in range(min(3, k))]
            right = [(i + 1, 0.0) for i in range(min(2, k))]
            if (n_segments + 1) - k - 1 == len(left) + len(right):
                bconditions = [(left, right)] * self.n_joints
            else:
                bconditions = [None] * self.n_joints
        self.bconditions = bconditions
        super().__init__(n_var, bounds)

    def _generate(self, x):
        return bspline_trajectory(
            x,
            num_joints=self.n_joints,
            k=self.k,
            waypoints=self.waypoints,
            bconditions=self.bconditions,
            steps=self.steps,
        )


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
