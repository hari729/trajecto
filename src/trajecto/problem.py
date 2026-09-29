import numpy as np

import pinocchio as pin
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.joblib import JoblibParallelization

from trajecto.urdf import (
    load_urdf_xml,
    inject_ft_sensors,
    parse_joint_limits,
    parse_joint_axes,
)


class RobotModel:
    def __init__(self, source, xacro_args=None):
        raw_urdf_xml = load_urdf_xml(source, xacro_args=xacro_args)
        self.model = pin.buildModelFromXML(raw_urdf_xml)
        self.joint_names = [
            self.model.names[i]
            for i in range(1, self.model.njoints)
            if self.model.joints[i].nq == 1 and self.model.joints[i].nv == 1
        ]
        self.urdf_xml = inject_ft_sensors(raw_urdf_xml, self.joint_names)
        # derive what the URDF provides: per-joint limits and rotation axes
        self.derived_limits = parse_joint_limits(raw_urdf_xml, self.joint_names)
        self.joint_axes = parse_joint_axes(raw_urdf_xml, self.joint_names)


class TrajectoryProblem(ElementwiseProblem):
    def __init__(
        self,
        trajectory_function,
        urdf_arg,
        n_var,
        bounds,
        trajectory_extras,
        joint_limits=None,
        time_limit=10.0,
        n_threads=4,
        **kwargs,
    ) -> None:
        self.trajectory_function = trajectory_function
        self.trajectory_extras = trajectory_extras
        self.urdf_arg = urdf_arg
        self.robotmodel = RobotModel(**urdf_arg)
        self.pin_model = self.robotmodel.model
        self.n_joints = self.pin_model.nv
        self.joint_limits = self._resolve_joint_limits(joint_limits)
        n_ieq_constr = (
            4 * self.n_joints + 1
        )  # time, velocity, acceleration, jerk, torque constraints
        self.joint_names = self.robotmodel.joint_names
        self.time_limit = time_limit
        self.n_threads = n_threads
        super().__init__(
            n_obj=3,
            n_var=n_var,
            n_ieq_constr=n_ieq_constr,
            xl=bounds[0, :],
            xu=bounds[1, :],
            elementwise_runner=JoblibParallelization(
                n_jobs=self.n_threads, backend="loky"
            ),
            **kwargs,
        )

    def _resolve_joint_limits(self, joint_limits):
        """Merge user overrides over the limits derived from the URDF.

        Velocity and torque default to the URDF <limit> values; acceleration
        and jerk have no URDF equivalent and must be supplied. Every key must
        be a per-joint array in URDF joint order.
        """
        derived = self.robotmodel.derived_limits
        overrides = dict(joint_limits or {})
        resolved = {}
        for key in ("velocity", "acceleration", "jerk", "torque"):
            if key in overrides:
                value = np.asarray(overrides[key], dtype=float)
            elif key in derived:
                value = np.asarray(derived[key], dtype=float)
            else:
                raise ValueError(
                    f"joint_limits is missing {key!r} and it cannot be derived "
                    "from the URDF; supply it (e.g. via RobotConfig.joint_limits)"
                )
            if value.shape != (self.n_joints,):
                raise ValueError(
                    f"joint_limits[{key!r}] must have shape ({self.n_joints},) "
                    f"(one value per movable joint in URDF order), got {value.shape}"
                )
            resolved[key] = value
        return resolved

    def _check_joint_count(self, trajectory):
        n = trajectory["position"].shape[1]
        if n != self.n_joints:
            raise ValueError(
                f"trajectory has {n} joints but the model has {self.n_joints} "
                f"({self.joint_names}); check waypoints/trajectory column order"
            )

    def _compute_torques(self, trajectory):
        pin_data = self.pin_model.createData()
        return np.array(
            [
                pin.rnea(
                    self.pin_model,
                    pin_data,
                    trajectory["position"][i],
                    trajectory["velocity"][i],
                    trajectory["acceleration"][i],
                )
                for i in range(len(trajectory["time"]))
            ]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        trajectory = self.trajectory_function(x, **self.trajectory_extras)
        self._check_joint_count(trajectory)
        torques = self._compute_torques(trajectory)

        duration = trajectory["time"][-1] - trajectory["time"][0]
        max_v = np.max(np.abs(trajectory["velocity"]), axis=0)
        max_a = np.max(np.abs(trajectory["acceleration"]), axis=0)
        max_j = np.max(np.abs(trajectory["jerk"]), axis=0)
        max_torque = np.max(np.abs(torques), axis=0)

        time_constr = duration - self.time_limit
        v_constr = max_v - self.joint_limits["velocity"]
        a_constr = max_a - self.joint_limits["acceleration"]
        j_constr = max_j - self.joint_limits["jerk"]
        torque_constr = max_torque - self.joint_limits["torque"]

        # instantaneous power per joint, shape (T, n_joints)
        power = torques * np.asarray(trajectory["velocity"])
        # ∫|τ·ω| dt per joint, summed over all joints
        E = np.sum(np.trapezoid(np.abs(power), trajectory["time"], axis=0))

        if duration <= 0:
            # degenerate trajectory: report worst-case objectives and let the
            # time constraint (duration - time_limit) drive infeasibility
            out["F"] = [duration, np.inf, np.inf]
            out["G"] = np.concatenate(
                ([time_constr], v_constr, a_constr, j_constr, torque_constr)
            )
            return

        int_jer = np.trapezoid(trajectory["jerk"] ** 2, trajectory["time"], axis=0)
        SJ = np.sum(np.sqrt(int_jer / duration))

        out["F"] = [duration, E, SJ]
        out["G"] = np.concatenate(
            ([time_constr], v_constr, a_constr, j_constr, torque_constr)
        )

    def generate_trajectory(self, x):
        trajectory = self.trajectory_function(x, **self.trajectory_extras)
        self._check_joint_count(trajectory)
        trajectory["joint_names"] = self.joint_names
        trajectory["torque"] = self._compute_torques(trajectory)
        return trajectory
