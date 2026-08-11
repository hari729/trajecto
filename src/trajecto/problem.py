# from datetime import time
import numpy as np

import pinocchio as pin
from pymoo.core.problem import ElementwiseProblem

from trajecto.urdf import load_urdf_xml, inject_ft_sensors


class RobotModel:
    def __init__(self, source, xacro_args=None):
        raw_urdf_xml = load_urdf_xml(source, xacro_args=xacro_args)  # step 1: resolve
        self.model = pin.buildModelFromXML(raw_urdf_xml)  # step 2: build from raw
        self.joint_names = [
            self.model.names[i]
            for i in range(1, self.model.njoints)
            if self.model.joints[i].nq == 1 and self.model.joints[i].nv == 1
        ]
        self.urdf_xml = inject_ft_sensors(
            raw_urdf_xml, self.joint_names
        )  # step 4: final, complete


class TrajectoryProblem(ElementwiseProblem):
    def __init__(
        self,
        trajectory_function,
        urdf_arg,
        n_var,
        bounds,
        trajectory_extras,
        joint_limits,
        time_limit=10.0,
        **kwargs,
    ) -> None:
        self.trajectory_function = trajectory_function
        self.trajectory_extras = trajectory_extras
        self.urdf_arg = urdf_arg
        self.robotmodel = RobotModel(**urdf_arg)
        self.joint_limits = joint_limits
        self.pin_model = self.robotmodel.model
        self.n_joints = self.pin_model.nv
        n_ieq_constr = (
            4 * self.n_joints + 1
        )  # time, velocity, acceleration, jerk, torque constraints
        self.joint_names = self.robotmodel.joint_names
        self.time_limit = time_limit
        super().__init__(
            n_obj=3,
            n_var=n_var,
            n_ieq_constr=n_ieq_constr,
            xl=bounds[0, :],
            xu=bounds[1, :],
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        trajectory = self.trajectory_function(x, **self.trajectory_extras)
        torques = []
        pin_data = self.pin_model.createData()
        for i in range(len(trajectory["time"])):
            tau = pin.rnea(
                self.pin_model,
                pin_data,
                trajectory["position"][i],
                trajectory["velocity"][i],
                trajectory["acceleration"][i],
            )
            torques.append(tau)

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

        power = (
            torques * trajectory["velocity"]
        )  # instantaneous power per joint, shape (T, n_joints)
        energy_per_joint = np.trapezoid(
            np.abs(power), trajectory["time"], axis=0
        )  # ∫|τ·ω| dt per joint
        E = np.sum(energy_per_joint)  # total energy across all joints

        int_jer = np.trapezoid(trajectory["jerk"] ** 2, trajectory["time"], axis=0)
        SJ = np.sum(np.sqrt(int_jer / duration))

        out["F"] = [duration, E, SJ]
        out["G"] = np.concatenate(
            ([time_constr], v_constr, a_constr, j_constr, torque_constr)
        )

    def generate_trajectory(self, x):
        trajectory = self.trajectory_function(x, **self.trajectory_extras)
        trajectory["joint_names"] = self.joint_names
        trajectory["torque"] = []
        pin_data = self.pin_model.createData()
        for i in range(len(trajectory["time"])):
            tau = pin.rnea(
                self.pin_model,
                pin_data,
                trajectory["position"][i],
                trajectory["velocity"][i],
                trajectory["acceleration"][i],
            )
            trajectory["torque"].append(tau)
        return trajectory
