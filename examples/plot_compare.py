from trajecto.plots import plot_trajectory_comparison
from pathlib import Path

plot_trajectory_comparison(
    str(Path(__file__).parent / "knee_trajectory.json"),
    str(Path(__file__).parent / "joint_states.json"),
    output_path="full_comparison.png",
)
