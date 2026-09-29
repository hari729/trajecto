"""trajecto — multi-objective trajectory optimization for robot manipulators.

Public API:

    from trajecto import (
        Pipeline,
        RobotConfig,
        WorldConfig,
        TrajectoryProblem,
        BSplineTrajectory,
        Trajectory,
    )
"""

from trajecto.config import RobotConfig, WorldConfig
from trajecto.orchestrator import Pipeline
from trajecto.problem import TrajectoryProblem
from trajecto.samples import BSplineTrajectory, Trajectory, bspline_trajectory

__all__ = [
    "BSplineTrajectory",
    "Pipeline",
    "RobotConfig",
    "Trajectory",
    "TrajectoryProblem",
    "WorldConfig",
    "bspline_trajectory",
]
