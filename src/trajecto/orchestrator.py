import numpy as np
from pathlib import Path
import os
import threading
import time

from trajecto.optimizer import merge_pareto_fronts, save_trajectory_results
from trajecto.problem import TrajectoryProblem
from trajecto.launch_sim import build_robot_launch
from trajecto.urdf import set_initial_joint_positions

from pymoo.optimize import minimize
from pymoo.parallelization.joblib import JoblibParallelization


from launch import LaunchService


class Pipeline:
    def __init__(
        self,
        robot_name,
        urdf_arg,
        waypoints,
        joint_limits,
        trajectory_generator,
        time_limit,
        n_var,
        var_bounds,
        trajectory_extras,
        algorithm,
        results_dir,
        seeds=[1],
        n_gen=100,
        n_threads=8,
    ):
        self.robot_name = robot_name
        self.results_dir = Path(results_dir / self.robot_name)
        os.makedirs(self.results_dir, exist_ok=True)
        self.urdf_arg = urdf_arg
        self.waypoints = waypoints
        self.joint_limits = joint_limits
        self.trajectory_generator = trajectory_generator
        self.var_bounds = var_bounds
        self.n_var = n_var
        self.trajectory_extras = trajectory_extras
        self.algorithm = algorithm
        self.seeds = seeds
        self.n_gen = n_gen
        self.n_threads = n_threads
        self.time_limit = time_limit

        # initialize the thread pool and create the runner
        runner = JoblibParallelization(n_jobs=self.n_threads, backend="loky")

        # define the problem by passing the starmap interface of the thread pool
        problem = TrajectoryProblem(
            urdf_arg=urdf_arg,
            trajectory_function=trajectory_generator,
            n_var=n_var,
            bounds=var_bounds,
            trajectory_extras=trajectory_extras,
            joint_limits=joint_limits,
            elementwise_runner=runner,
            time_limit=time_limit,
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
        X = self.final_front.get("X")

        print(f"Global Pareto front: {len(F)} solutions")
        print(f"Objectives [Duration, Energy, Jerk]:\n{F}")

        save_trajectory_results(self.final_front, self.results_dir, self.problem)
        print(f"Results saved to {self.results_dir}")

    def launch_simulation(self, controllers_yaml_path, world_file, world_name):
        print("Launching simulation...")

        controller_name = "joint_trajectory_controller"

        ld = build_robot_launch(
            robot_model=self.problem.robotmodel,
            controllers_yaml_path=controllers_yaml_path,
            controller_name=controller_name,
            robot_name=self.robot_name,
            world_name=world_name,
            world_file=world_file,
        )
        self.launch_service = LaunchService()
        self.launch_service.include_launch_description(ld)
        self.launch_thread = threading.Thread(
            target=self.launch_service.run, daemon=True
        )
        self.launch_thread.start()

        # spawn/controller activation isn't instant — give it time before
        # simulate_trajectory tries to send a goal. A fixed sleep is crude
        # but workable; polling controller_manager's list_controllers
        # service would be more robust if goals start failing intermittently.
        import time

        time.sleep(8)

        self.simul_results_dir = Path(self.results_dir / f"{world_name}")
        os.makedirs(self.simul_results_dir, exist_ok=True)

    def shutdown_simulation(self):
        if hasattr(self, "launch_service"):
            self.launch_service.shutdown()
            self.launch_thread.join(timeout=5)

    # def simulate_trajectory(self, trajectory_name, show=False):
    #     from rclpy.executors import MultiThreadedExecutor
    #     from trajecto.nodes import publish_trajectory, record_joint_states
    #     import rclpy
    #
    #     rclpy.init()
    #
    #     node = publish_trajectory(
    #         self.results_dir / f"{trajectory_name}-trajectory.json"
    #     )
    #     recorder = record_joint_states(
    #         [
    #             str(self.simul_results_dir / f"{trajectory_name}-joint-states.json"),
    #             str(self.simul_results_dir / f"{trajectory_name}-ft-sensor.json"),
    #         ],
    #         joint_names=self.problem.robotmodel.joint_names,
    #         robot_name=self.robot_name,
    #     )
    #
    #     executor = MultiThreadedExecutor()
    #     executor.add_node(node)
    #     executor.add_node(recorder)
    #
    #     result = node.send_trajectory(executor)
    #     recorder.save_readings()
    #
    #     node.destroy_node()
    #     recorder.destroy_node()
    #     rclpy.shutdown()
    #
    #     if show:
    #         from trajecto.plots import plot_rnea_on_measured
    #
    #         plot_rnea_on_measured(
    #             str(
    #                 self.results_dir / f"{trajectory_name}-trajectory.json"
    #             ),  # confirm actual trajectory filename from save_trajectory_results
    #             str(self.simul_results_dir / f"{trajectory_name}-joint-states.json"),
    #             str(self.simul_results_dir / f"{trajectory_name}-ft-sensor.json"),
    #             rmodel=self.problem.robotmodel,
    #         )

    def run_simulation(
        self,
        controllers_yaml_path,
        world_file,
        world_name,
        trajectory_name,
        startup_wait=8.0,
    ):
        print("Launching simulation...")
        import json

        with open(
            self.results_dir / f"{trajectory_name}-trajectory.json"
        ) as f:  # confirm actual filename
            traj_data = json.load(f)
        start_q = traj_data["q"][0]

        launch_urdf_xml = set_initial_joint_positions(
            self.problem.robotmodel.urdf_xml,
            self.problem.robotmodel.joint_names,
            start_q,
        )
        self.problem.robotmodel.urdf_xml = launch_urdf_xml

        controller_name = "joint_trajectory_controller"
        ld = build_robot_launch(
            robot_model=self.problem.robotmodel,
            controllers_yaml_path=controllers_yaml_path,
            controller_name=controller_name,
            robot_name=self.robot_name,
            world_name=world_name,
            world_file=world_file,
        )
        self.launch_service = LaunchService()
        self.launch_service.include_launch_description(ld)

        self.simul_results_dir = Path(self.results_dir / f"{world_name}")
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
            self.results_dir / f"{trajectory_name}-trajectory.json"
        )
        recorder = record_joint_states(
            [
                str(self.simul_results_dir / f"{trajectory_name}-joint-states.json"),
                str(self.simul_results_dir / f"{trajectory_name}-ft-sensor.json"),
            ],
            joint_names=self.problem.robotmodel.joint_names,  # fixed from last message's #3
            robot_name=self.robot_name,
        )

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(recorder)

        node.send_trajectory(executor)
        recorder.save_readings()

        node.destroy_node()
        recorder.destroy_node()
        rclpy.shutdown()
