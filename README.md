# trajecto

Multi-objective trajectory optimization for robot manipulators, built on
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) for rigid-body
dynamics and [pymoo](https://pymoo.org/) for the multi-objective solver.

The optimization problem minimizes three objectives — trajectory **duration**,
**energy** (`∫|τ·ω| dt`), and **smoothness** (squared-jerk integral) — subject
to time, velocity, acceleration, jerk, and torque constraints.

## Installation

trajecto is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Requires Python >= 3.11.

## Usage

See [examples/example.py](examples/example.py) for the end-to-end pipeline on
a UR5: define joint-space waypoints, pick a trajectory generator from
[`trajecto.samples`](src/trajecto/samples.py) (`bspline_trajectory`),
configure it via `trajectory_extras`, set `joint_limits`, then build a
`Pipeline` to optimize and run the trajectory in a Gazebo simulation:

```python
pipe = Pipeline(
    robot_name="ur",
    urdf_arg=URDF_PATH,
    waypoints=waypoints,
    trajectory_generator=bspline_trajectory,
    trajectory_extras=trajectory_extras,
    joint_limits=joint_limits,
    time_limit=50,
    n_var=5,
    var_bounds=...,
    algorithm=MO_BWR(pop_size=100),
    results_dir=...,
)
pipe.optimize()
pipe.run_simulation(...)
```

### The trajectory function contract

A trajectory generator is any callable `trajectory_function(x, **trajectory_extras)`
that returns a dict containing:

| key            | shape        | meaning                  |
| -------------- | ------------ | ------------------------ |
| `time`         | `(T,)`       | time stamps              |
| `position`     | `(T, n_joints)` | joint positions      |
| `velocity`     | `(T, n_joints)` | joint velocities     |
| `acceleration` | `(T, n_joints)` | joint accelerations  |
| `jerk`         | `(T, n_joints)` | joint jerks          |

`TrajectoryProblem.generate_trajectory(x)` returns the same dict with
`joint_names` and `torque` (from `pin.rnea`) added.

## Joint ordering contract

**All joint-ordered inputs must be supplied in URDF joint order** — i.e. the
order in which the movable joints are declared in the URDF:

- the **columns** of `waypoints`,
- the **columns** of the trajectory arrays returned by the trajectory
  function (`position`, `velocity`, `acceleration`, `jerk`),
- the per-joint entries of `joint_limits` (`velocity`, `acceleration`,
  `jerk`, `torque`).

Pinocchio's `buildModelFromXML` extracts joints deterministically in URDF
declaration order, and `trajecto` does **not** reorder anything: column `i` of
every trajectory array is assumed to correspond one-to-one to
`joint_names[i]` (see `TrajectoryProblem.joint_names`, which mirrors
`RobotModel.joint_names`). It is the user's responsibility to provide
waypoints, trajectory extras, and limits in that order.

To inspect the expected order for a given URDF:

```python
print(problem.joint_names)
# e.g. ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
```

`tests/test_problem.py` asserts the UR5 joint order and that trajectory
columns follow the waypoint/model order.

## Tests

```bash
uv run pytest
```
