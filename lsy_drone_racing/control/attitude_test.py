"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are dynamically updated based on actual gate and obstacle positions detected during flight.
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
        self.drone_mass = drone_params["mass"]

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.g = 9.81

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

        # Original waypoints designed for nominal gate positions
        self._original_waypoints = np.array([
            [-1.5, 0.75, 0.05],   # 0: Start position
            [-1.0, 0.55, 0.4],    # 1: Approaching Gate 0
            [0.3, 0.35, 0.7],     # 2: Through Gate 0 (nominal)
            [1.3, -0.15, 0.9],    # 3: Approaching Gate 1
            [0.85, 0.85, 1.2],    # 4: Through Gate 1 (nominal)
            [-0.5, -0.05, 0.7],   # 5: Approaching Gate 2
            [-1.2, -0.2, 0.8],   # 6: Through Gate 2 (nominal)
            [-1.2, -0.2, 1.2],    # 7: Rising up, approaching Gate 3
            [0.0, -0.7, 1.2],    # 8: Through Gate 3 (nominal)
            [0.5, -0.75, 1.2],    # 9: End position
        ])
        
        # Map waypoints to gates (waypoint_index: gate_index)
        self._waypoint_gate_map = {
            2: 0,  # Waypoint 2 passes through Gate 0
            4: 1,  # Waypoint 4 passes through Gate 1
            6: 2,  # Waypoint 6 passes through Gate 2
            8: 3,  # Waypoint 8 passes through Gate 3
        }
        
        # Current waypoints (will be updated based on actual positions)
        self._current_waypoints = self._original_waypoints.copy()
        
        # Smooth update parameters
        self._waypoint_update_rate = 0.5  # Update rate per step (50% per step for faster multi-waypoint response)
        self._waypoint_targets = {}  # Target positions for waypoints being updated
        
        # Track which objects have been detected
        self._detected_gates = set()
        self._detected_obstacles = set()
        self._updated_obstacles = []
        
        # Obstacle avoidance parameters
        self._obstacle_radius = 0.7  # Increased: detect obstacles earlier
        self._avoidance_gain = 0.3  # Increased: stronger repulsion force
        self._max_avoidance_offset = 0.2  # Increased: allow larger corrections

        self._t_total = 15  # s
        self._update_trajectory()
        
        self._tick = 0
        self._finished = False

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

    def _update_trajectory(self):
        """Regenerate the trajectory spline based on current waypoints."""
        t = np.linspace(0, self._t_total, len(self._current_waypoints))
        self._des_pos_spline = CubicSpline(t, self._current_waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

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
        if t >= self._t_total:
            self._finished = True

        # Smoothly update waypoints towards their targets
        trajectory_updated = False
        for wp_idx, target_pos in list(self._waypoint_targets.items()):
            current_pos = self._current_waypoints[wp_idx]
            diff = target_pos - current_pos
            distance = np.linalg.norm(diff)
            
            # If close enough to target, snap to it and remove from targets
            if distance < 0.01:
                self._current_waypoints[wp_idx] = target_pos
                del self._waypoint_targets[wp_idx]
                trajectory_updated = True
            else:
                # Move towards target by update_rate
                self._current_waypoints[wp_idx] += self._waypoint_update_rate * diff
                trajectory_updated = True
        
        # Regenerate trajectory if waypoints were updated
        if trajectory_updated:
            self._update_trajectory()

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Apply obstacle avoidance
        if len(self._updated_obstacles) > 0:
            avoidance_vector = np.zeros(3)
            
            for obs_pos in self._updated_obstacles:
                diff = des_pos - obs_pos
                distance = np.linalg.norm(diff)
                
                if distance < self._obstacle_radius:
                    repulsion = self._avoidance_gain * (1.0 / distance - 1.0 / self._obstacle_radius)
                    avoidance_vector += repulsion * (diff / distance)
            
            # Limit avoidance magnitude
            avoidance_magnitude = np.linalg.norm(avoidance_vector)
            if avoidance_magnitude > self._max_avoidance_offset:
                avoidance_vector = avoidance_vector / avoidance_magnitude * self._max_avoidance_offset
            
            des_pos = des_pos + avoidance_vector

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
        """Increment the tick counter and update waypoints based on detected objects.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1

        # Check gates in sensor range and update corresponding waypoints
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
                            print(f"  Detection distance: {distance:.3f}m")
                            
                            # Get nominal gate orientation
                            nominal_rpy = self._nominal_gates[i]['rpy']
                            
                            # Get gate orientation
                            if 'gates_quat' in obs:
                                actual_quat = obs["gates_quat"][i]
                                # Convert quaternion to rotation matrix and euler angles
                                rot = R.from_quat(actual_quat)
                                actual_rpy = rot.as_euler('xyz')
                                rpy_diff = actual_rpy - nominal_rpy
                                print(f"  Nominal rpy: {nominal_rpy}")
                                print(f"  Actual rpy:  {actual_rpy}")
                                print(f"  RPY diff:    {rpy_diff}")
                                
                                # Calculate gate normal vector based on actual orientation
                                # For vertical gates, X-axis points through the gate (forward direction)
                                gate_normal = rot.as_matrix()[:, 0]  # X-axis is the direction through gate
                                print(f"  Gate forward direction: {gate_normal}")
                            else:
                                # Fallback: use nominal orientation
                                rot = R.from_euler('xyz', nominal_rpy)
                                gate_normal = rot.as_matrix()[:, 0]  # X-axis
                                print(f"  WARNING: No gates_quat in obs, using nominal orientation!")
                            
                            # Update the corresponding waypoint and adjacent waypoints
                            for wp_idx, gate_idx in self._waypoint_gate_map.items():
                                if gate_idx == i:
                                    # Gate center waypoint: use actual gate position
                                    target_waypoint = gate_pos.copy()
                                    self._waypoint_targets[wp_idx] = target_waypoint
                                    print(f"  -> Waypoint {wp_idx} (gate center): {target_waypoint}")
                                    
                                    # Approach waypoint: maintain original distance but adjust direction
                                    if wp_idx > 0:
                                        prev_wp_idx = wp_idx - 1
                                        # Calculate original approach distance and direction
                                        original_approach = self._original_waypoints[prev_wp_idx]
                                        original_gate = self._nominal_gates[gate_idx]['pos']
                                        original_offset = original_approach - original_gate
                                        approach_distance = np.linalg.norm(original_offset[:2])  # Horizontal distance
                                        approach_distance = min(approach_distance, 1.2)  # Limit max distance
                                        
                                        # Position along gate normal at similar distance, preserve height
                                        prev_target = gate_pos.copy()
                                        prev_target[:2] -= approach_distance * gate_normal[:2]  # Horizontal offset
                                        prev_target[2] = original_approach[2]  # Keep original height
                                        self._waypoint_targets[prev_wp_idx] = prev_target
                                        print(f"  -> Waypoint {prev_wp_idx} (approach): {approach_distance:.2f}m before gate, height={prev_target[2]:.2f}")
                                    
                                    # Exit waypoint: maintain original distance but adjust direction
                                    if wp_idx < len(self._original_waypoints) - 1:
                                        next_wp_idx = wp_idx + 1
                                        # Calculate original exit distance and direction
                                        original_exit = self._original_waypoints[next_wp_idx]
                                        original_gate = self._nominal_gates[gate_idx]['pos']
                                        original_offset = original_exit - original_gate
                                        exit_distance = np.linalg.norm(original_offset[:2])  # Horizontal distance
                                        exit_distance = min(exit_distance, 1.2)  # Limit max distance
                                        
                                        # Position along gate normal at similar distance, preserve height
                                        next_target = gate_pos.copy()
                                        next_target[:2] += exit_distance * gate_normal[:2]  # Horizontal offset
                                        next_target[2] = original_exit[2]  # Keep original height
                                        self._waypoint_targets[next_wp_idx] = next_target
                                        print(f"  -> Waypoint {next_wp_idx} (exit): {exit_distance:.2f}m after gate, height={next_target[2]:.2f}")
                                    
                                    break

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
                            self._updated_obstacles.append(obstacle_pos.copy())
                            
                            # Proactively adjust nearby waypoints to avoid obstacle
                            for wp_idx in range(len(self._current_waypoints)):
                                wp_pos = self._current_waypoints[wp_idx]
                                dist_to_obstacle = np.linalg.norm(wp_pos - obstacle_pos)
                                
                                # If waypoint is dangerously close to obstacle (within 0.4m)
                                if dist_to_obstacle < 0.4:
                                    # Push waypoint away from obstacle
                                    push_direction = (wp_pos - obstacle_pos) / dist_to_obstacle
                                    push_amount = 0.4 - dist_to_obstacle
                                    adjusted_wp = wp_pos + push_direction * push_amount
                                    self._waypoint_targets[wp_idx] = adjusted_wp
                                    print(f"  -> Waypoint {wp_idx} too close! Pushing {push_amount:.2f}m away")
                        else:
                            self._updated_obstacles.append(obstacle_pos.copy())

        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
        self._detected_gates.clear()
        self._detected_obstacles.clear()
        self._updated_obstacles.clear()
        self._waypoint_targets.clear()
        self._current_waypoints = self._original_waypoints.copy()
        self._update_trajectory()