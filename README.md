# Autonomous Drone Racing Project Course

<p align="center">
  <img width="460" height="300" src="docs/img/banner.jpeg">
</p>
<sub><sup>AI-generated image</sup></sub>

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Documentation Status]][Documentation Status URL] [![Tests]][Tests URL]

---

## Introduction

**LSY Drone Racing** is a course project designed to help you develop and evaluate autonomous drone racing algorithms — both in simulation and on real Crazyflie hardware.
Whether you’re new to drones or an experienced developer, this project provides a structured and practical way to explore high-speed autonomy, control, and perception in dynamic environments.

---

## Documentation

To get started, visit our [official documentation](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html).

---

## Dependencies

This project builds upon several open-source packages developed by the [Learning Systems Lab (LSY)](https://www.ce.cit.tum.de/lsy/home/) at TUM.You can explore these related projects:

- [**crazyflow**](https://github.com/utiasDSL/crazyflow) – A high-speed, high-fidelity drone simulator with strong sim-to-real performance.
- [**drone-models**](https://github.com/utiasDSL/drone-models) – A collection of accurate drone models for simulation and model-based control.
- [**drone-controllers**](https://github.com/utiasDSL/drone-controllers) – Controllers for the Crazyflie quadrotor.

---

## Additional Scripts and Experimental Extensions

### New Scripts

This project includes several implementations for drone racing:

#### 1. **Path Planning Module** (`path_planning.py`)

A comprehensive path planning system that supports:

- Intelligent waypoint generation considering gate orientations
- Backtracking detection and automatic detour planning
- Obstacle avoidance with configurable safety margins
- Arc-length reparameterization for smooth trajectory following
- Trajectory extension for stable end-of-track behavior

#### 2. **MPCC Controller** (`mpcc_controller.py`)

A Model Predictive Contouring Control (MPCC) controller featuring:

- Arc-length parameterized trajectory following using acados optimization
- Modular path planning integration via the path planning module
- Dynamic replanning when environment changes are detected
- Configurable speed/stability trade-offs through tunable cost function weights
- Robust handling of randomized inertial properties and obstacles

#### 3. **State-Based Controller** (`state_test.py`) - *Early Version*
An early trajectory-following controller for learning and testing purposes:
- Dynamic replanning when environment changes are detected

> **Note:** This is an early-stage controller used for initial development and testing. For optimal production use and performance, refer to the MPCC controller.

#### 4. **Trajectory Visualizer** (`trajectory_visualizer.py`)

A visualization tool for analyzing flight performance:

- Records position, velocity, and instantaneous speed during flight
- Generates trajectory plots (XY and XZ views) colored by speed
- Multi-episode support with automatic timestamped folder creation
- Dynamic gate/obstacle position updates based on sensor range
- Saves data in both image (.png) and raw data (.npz) formats

### Advanced Simulation Script (`sim_traj.py`)

The simulation script has been extended with several new features:

1. **Real-time Trajectory Visualization**

   - Draws the planned trajectory in green during simulation
   - Updates dynamically as the controller replans
   - Automatically reduces line complexity to maintain performance
2. **Performance Statistics**

   - Tracks success count across multiple runs
   - Calculates average completion time for successful runs
   - Provides detailed summary after all episodes complete
3. **Enhanced Visualization Mode**

   - Activated using the `-v` flag
   - Generates detailed trajectory plots after each episode
   - Saves visualization data for post-analysis

#### Usage Example

```bash
# Run with trajectory visualization enabled
python scripts/sim_traj.py --config level2.toml --controller mpcc_controller.py -r -v

# Run multiple episodes for statistics
python scripts/sim_traj.py --config level2.toml --controller mpcc_controller.py --n_runs 20
```

**Command-line options:**

- `--config`: Configuration file (e.g., `level0.toml`, `level2.toml`);
- `--controller`: Controller file to use (e.g., `mpcc_controller.py`);
- `--n_runs`: Number of episodes to run;
- `-r` or `--render`: Enable GUI;
- `-v`: Enable trajectory visualization.

### Performance Results

The MPCC controller has been tested extensively on Level 2 and Level 3 configurations with various horizon parameters:


| Horizon Steps | μ   | Level-2 Success Rate | Level-2 Avg. Time (s) | Level-3 Success Rate | Level-3 Avg. Time (s) |
| ------------- | ---- | -------------------- | --------------------- | -------------------- | --------------------- |
| 40            | 10.0 | 85%                  | 6.28                  | 90%                  | 8.02                  |
|               | 14.0 | 70%                  | 5.94                  | 80%                  | 7.88                  |
|               | 18.0 | 65%                  | 5.77                  | 70%                  | 7.84                  |
| 35            | 10.0 | 65%                  | 5.58                  | 80%                  | 7.30                  |
|               | 14.0 | 60%                  | 5.13                  | 75%                  | 7.01                  |
|               | 18.0 | 50%                  | 5.02                  | 70%                  | 6.89                  |

**TABLE I: MPCC controller performance in Level 2 and Level 3 (20 runs per configuration)**

### Demo Videos

Real-world flight demonstrations during lab session:

https://github.com/user-attachments/assets/b6926c07-d23c-4314-8041-42394461c99a

**VIDEO I: MPCC controller performance in Level 2**


https://github.com/user-attachments/assets/f2c8e4f3-2c19-438b-92ea-3dbaa9eb80c7

**VIDEO II: MPCC controller performance in Level 3**

---

## Difficulty Levels

Each task setup — from track design to physics configuration — is defined by a TOML file (e.g., [`level0.toml`](config/level0.toml)).
The configuration files specify progressive difficulty levels from easy (0) to hard (3):


|      Evaluation Scenario      | Rand. Inertial Properties | Randomized Obstacles, Gates | Random Tracks |             Notes             |
| :---------------------------: | :-----------------------: | :-------------------------: | :-----------: | :----------------------------: |
| [Level 0](config/level0.toml) |           *No*           |            *No*            |     *No*     |       Perfect knowledge       |
| [Level 1](config/level1.toml) |          **Yes**          |            *No*            |     *No*     |        Adaptive control        |
| [Level 2](config/level2.toml) |          **Yes**          |           **Yes**           |     *No*     |          Re-planning          |
| [Level 3](config/level3.toml) |          **Yes**          |           **Yes**           |    **Yes**    |        Online planning        |
|         **sim2real**         |     **Real hardware**     |           **Yes**           |    **Yes**    | Simulation-to-reality transfer |

---

## Online Competition

Throughout the semester, teams will compete to achieve the fastest autonomous race completion times.Competition results are hosted on Kaggle — a popular machine learning competition platform.

> **Note:** Competition results **do not** directly affect your course grade.
> However, they provide valuable feedback on the performance and robustness of your approach compared to others.

The competition environment always uses **difficulty level 2**.
If your code fails the automated tests, it is likely to encounter the same issues in our evaluation environment.
For full details, refer to the [documentation](https://lsy-drone-racing.readthedocs.io/en/latest/).

---

## File Structure

```
lsy_drone_racing/
├── control/
│   ├── path_planning.py            # Path planning module
│   ├── mpcc_controller.py          # MPCC controller implementation
│   ├── state_test.py               # Early trajectory-following controller
│   └── ...
├── utils/
│   ├── trajectory_visualizer.py    # Trajectory visualization tool
│   └── ...
├── scripts/
│   ├── sim_traj.py                 # Enhanced simulation script
│   └── ...
└── config/
    ├── level0.toml                 # Configuration files
    ├── level1.toml
    ├── level2.toml
    └── level3.toml
```
---
