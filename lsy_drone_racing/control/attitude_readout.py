"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeController(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._sensor_range = config.env.sensor_range

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

        # Same waypoints as in the position controller. Determined by trial and error.
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self._tick = 0
        self._finished = False

        # Store nominal positions from config
        self._nominal_gates = []
        for gate in config.env.track.gates:
            gate_info = {
                'pos': np.array(gate['pos']),
                'rpy': np.array(gate['rpy'])
            }
            self._nominal_gates.append(gate_info)

        self._nominal_obstacles = []
        for obstacle in config.env.track.obstacles:
            obstacle_info = {
                'pos': np.array(obstacle['pos'])
            }
            self._nominal_obstacles.append(obstacle_info)

        # Track which objects have been detected and updated
        self._detected_gates = set()
        self._detected_obstacles = set()

        # Print nominal positions at initialization
        print("\n" + "="*60)
        print("NOMINAL GATE POSITIONS AND ORIENTATIONS:")
        print("="*60)
        for i, gate in enumerate(self._nominal_gates):
            print(f"Gate {i}: pos={gate['pos']}, rpy={gate['rpy']}")
        
        print("\n" + "="*60)
        print("NOMINAL OBSTACLE POSITIONS:")
        print("="*60)
        for i, obstacle in enumerate(self._nominal_obstacles):
            print(f"Obstacle {i}: pos={obstacle['pos']}")
        print("="*60 + "\n")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter and check for objects entering sensor range.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1

        # Check gates in sensor range
        if 'gates_pos' in obs:
            for i, gate_pos in enumerate(obs['gates_pos']):
                if i not in self._detected_gates:
                    distance = np.linalg.norm(gate_pos - obs['pos'])
                    if distance <= self._sensor_range:
                        self._detected_gates.add(i)
                        nominal_pos = self._nominal_gates[i]['pos']
                        pos_diff = gate_pos - nominal_pos
                        
                        if np.linalg.norm(pos_diff) > 1e-6:
                            print(f"\n[GATE {i}] Position changed!")
                            print(f"  Nominal pos: {nominal_pos}")
                            print(f"  Actual pos:  {gate_pos}")
                            print(f"  Difference:  {pos_diff}")

        # Check obstacles in sensor range
        if 'obstacles_pos' in obs:
            for i, obstacle_pos in enumerate(obs['obstacles_pos']):
                if i not in self._detected_obstacles:
                    distance = np.linalg.norm(obstacle_pos - obs['pos'])
                    if distance <= self._sensor_range:
                        self._detected_obstacles.add(i)
                        nominal_pos = self._nominal_obstacles[i]['pos']
                        pos_diff = obstacle_pos - nominal_pos
                        
                        if np.linalg.norm(pos_diff) > 1e-6:
                            print(f"\n[OBSTACLE {i}] Position changed!")
                            print(f"  Nominal pos: {nominal_pos}")
                            print(f"  Actual pos:  {obstacle_pos}")
                            print(f"  Difference:  {pos_diff}")

        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
        self._detected_gates.clear()
        self._detected_obstacles.clear()