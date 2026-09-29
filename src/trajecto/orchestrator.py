import json
from pathlib import Path
import os
import threading
import time

from trajecto.config import RobotConfig, WorldConfig
from trajecto.optimizer import merge_pareto_fronts, save_trajectory_results
from trajecto.problem import TrajectoryProblem
from trajecto.launch_sim import build_robot_launch
from trajecto.urdf import set_initial_joint_positions

from pymoo.optimize import minimize


class Pipeline:
    """Optimize a trajectory for a robot, then simulate it in Gazebo.

    robot:   RobotConfig describing the robot (URDF, controllers, limits).
    world:   optional WorldConfig — required only for run_simulation().
    trajectory_generator / trajectory_extras / n_var / var_bounds: the
        trajectory model and its optimization variables (unchanged contract).
    joint_limits: optional overrides merged over the limits derived from the
        robot URDF (velocity/torque). If None, robot.joint_limits is used.
        Acceleration and jerk have no URDF equivalent and must be supplied via
        one of these.
    """

    def __init__(
        self,
        robot,
        trajectory_generator,
        n_var,
        var_bounds,
        trajectory_extras,
        algorithm,
        results_dir,
        time_limit,
        joint_limits=None,
        world=None,
        seeds=(1,),
        n_gen=100,
        n_threads=8,
    ):
        if not isinstance(robot, RobotConfig):
            raise TypeError(
                f"Pipeline robot must be a RobotConfig, got {type(robot).__name__}"
            )
        if world is not None and not isinstance(world, WorldConfig):
            raise TypeError(
                f"Pipeline world must be a WorldConfig, got {type(world).__name__}"
            )
        self.robot = robot
        self.world = world
        self.robot_name = robot.name
        self.results_dir = Path(results_dir) / self.robot_name
        os.makedirs(self.results_dir, exist_ok=True)
        if joint_limits is None:
            joint_limits = robot.joint_limits
        self.joint_limits = joint_limits
        self.trajectory_generator = trajectory_generator
        self.var_bounds = var_bounds
        self.n_var = n_var
        self.trajectory_extras = trajectory_extras
        self.algorithm = algorithm
        self.seeds = seeds
        self.n_gen = n_gen
        self.time_limit = time_limit

        problem = TrajectoryProblem(
            urdf_arg={"source": robot.urdf_source, "xacro_args": robot.xacro_args},
            trajectory_function=trajectory_generator,
            n_var=n_var,
            bounds=var_bounds,
            trajectory_extras=trajectory_extras,
            joint_limits=joint_limits,
            time_limit=time_limit,
            n_threads=n_threads,
        )
        self.problem = problem

    def optimize(self):

        print("Running multi-objective optimization...")
        results = []
        for seed in self.seeds:
            # execute the optimization
            res = minimize(
                self.problem,
                self.algorithm,
                termination=("n_gen", self.n_gen),
                seed=seed,
            )
            results.append(res)
            print("ExecTime:", res.exec_time)

        self.final_front = merge_pareto_fronts(results)

        F = self.final_front.get("F")

        print(f"Global Pareto front: {len(F)} solutions")
        print(f"Objectives [Duration, Energy, Jerk]:\n{F}")

        save_trajectory_results(self.final_front, self.results_dir, self.problem)
        print(f"Results saved to {self.results_dir}")

    def run_simulation(
        self,
        trajectory_name,
        world=None,
        startup_wait=8.0,
        headless=False,
    ):
        """Run a saved trajectory in Gazebo and record joint states + FT data.

        world: WorldConfig; defaults to the one given at construction.
        headless: run the Gazebo server without the GUI — use on machines or
            containers without a display. Recording and plots are unaffected.
        """
        world = world or self.world
        if world is None:
            raise ValueError(
                "run_simulation needs a WorldConfig — pass it to Pipeline(world=...)"
                " or to run_simulation(world=...)"
            )
        print("Launching simulation...")
        from launch import LaunchService

        with open(self.results_dir / f"{trajectory_name}-trajectory.json") as f:
            traj_data = json.load(f)
        start_q = traj_data["position"][0]

        launch_urdf_xml = set_initial_joint_positions(
            self.problem.robotmodel.urdf_xml,
            self.problem.robotmodel.joint_names,
            start_q,
        )

        ld = build_robot_launch(
            robot_model=self.problem.robotmodel,
            urdf_xml=launch_urdf_xml,
            controllers_yaml_path=self.robot.controllers_yaml,
            controller_name=self.robot.controller_name,
            robot_name=self.robot_name,
            world_name=world.name,
            world_file=world.sdf_path,
            headless=headless,
        )
        self.launch_service = LaunchService()
        self.launch_service.include_launch_description(ld)

        self.simul_results_dir = Path(self.results_dir / f"{world.name}")
        os.makedirs(self.simul_results_dir, exist_ok=True)

        def _run_trajectory_and_shutdown():
            time.sleep(startup_wait)  # crude wait for controllers to spawn/activate
            self._simulate_trajectory(trajectory_name)
            self.launch_service.shutdown()

        worker = threading.Thread(target=_run_trajectory_and_shutdown, daemon=True)
        worker.start()

        self.launch_service.run()  # blocks main thread until shutdown() above fires
        worker.join()

    def _simulate_trajectory(self, trajectory_name):
        from rclpy.executors import MultiThreadedExecutor
        from trajecto.nodes import publish_trajectory, record_joint_states
        import rclpy

        rclpy.init()
        node = publish_trajectory(
            self.results_dir / f"{trajectory_name}-trajectory.json",
            controller_name=self.robot.controller_name,
        )
        recorder = record_joint_states(
            [
                str(self.simul_results_dir / f"{trajectory_name}-joint-states.json"),
                str(self.simul_results_dir / f"{trajectory_name}-ft-sensor.json"),
            ],
            joint_names=self.problem.robotmodel.joint_names,
            robot_name=self.robot_name,
            joint_axes=self.problem.robotmodel.joint_axes,
        )

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(recorder)

        node.send_trajectory(executor)
        recorder.save_readings()

        node.destroy_node()
        recorder.destroy_node()
        rclpy.shutdown()
