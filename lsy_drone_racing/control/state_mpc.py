"""MPC-based Local Avoidance Controller.

This controller maintains the planned waypoints but uses Model Predictive Control
to perform local collision avoidance in the XY plane, keeping Z-axis stable for gate passage.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPCAvoidanceController(Controller):
    """Controller using MPC for real-time local obstacle avoidance."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the MPC avoidance controller."""
        super().__init__(obs, info, config)
        
        # Environment parameters
        self._freq = config.env.freq
        self._sensor_range = config.env.sensor_range
        self._dt = 1.0 / self._freq
        
        # Drone parameters
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81
        
        # PID gains
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        
        # Store nominal positions
        self._nominal_gates = []
        for gate in config.env.track.gates:
            self._nominal_gates.append({
                'pos': np.array(gate['pos']),
                'rpy': np.array(gate['rpy'])
            })
        
        self._nominal_obstacles = []
        for obstacle in config.env.track.obstacles:
            self._nominal_obstacles.append({
                'pos': np.array(obstacle['pos'])
            })
        
        # MPC parameters
        self._mpc_horizon = 15  # prediction steps (0.3 seconds ahead)
        self._mpc_sample_directions = 12  # number of directions to sample
        self._mpc_sample_distances = [0.05, 0.10, 0.15, 0.20]  # meters
        
        # Obstacle avoidance parameters
        self._obstacle_influence_radius = 0.7  # meters
        self._obstacle_danger_radius = 0.3     # meters
        self._gate_corridor_width = 0.4        # safe width for gate passage
        self._gate_corridor_length = 1.5       # length of gate corridor
        
        # Tracking parameters
        self._detected_gates = {}
        self._detected_obstacles = {}
        self._active_gate_idx = None  # currently passing through this gate
        
        # Waypoint updates
        self._waypoint_update_rate = 0.6
        self._waypoint_targets = {}
        
        # Initial waypoints
        self._original_waypoints = self._generate_waypoints()
        self._current_waypoints = self._original_waypoints.copy()
        
        # Trajectory
        self._t_total = 30.0
        self._update_trajectory()
        
        # State
        self._tick = 0
        self._finished = False
        
        print("\n" + "="*60)
        print("MPC LOCAL AVOIDANCE CONTROLLER INITIALIZED")
        print("="*60)
        print(f"MPC horizon: {self._mpc_horizon} steps ({self._mpc_horizon * self._dt:.2f}s)")
        print(f"Sample directions: {self._mpc_sample_directions}")
        print(f"Initial waypoints: {len(self._original_waypoints)}")
        print("="*60 + "\n")

    def _generate_waypoints(self) -> np.ndarray:
        """Generate initial waypoints."""
        waypoints = []
        waypoints.append(np.array([-1.5, 0.75, 0.05]))
        
        for gate in self._nominal_gates:
            gate_pos = gate['pos']
            gate_rpy = gate['rpy']
            rot = R.from_euler('xyz', gate_rpy)
            gate_normal = rot.as_matrix()[:, 0]
            
            # Approach
            approach = gate_pos - 0.8 * gate_normal
            approach[2] = gate_pos[2]
            waypoints.append(approach)
            
            # Center
            waypoints.append(gate_pos.copy())
            
            # Exit
            exit_pt = gate_pos + 0.6 * gate_normal
            exit_pt[2] = gate_pos[2]
            waypoints.append(exit_pt)
        
        # Final
        waypoints.append(waypoints[-1] + np.array([0.3, 0.0, 0.0]))
        
        return np.array(waypoints)

    def _update_trajectory(self):
        """Regenerate trajectory spline."""
        n = len(self._current_waypoints)
        t = np.linspace(0, self._t_total, n)
        self._des_pos_spline = CubicSpline(t, self._current_waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._des_acc_spline = self._des_vel_spline.derivative()

    def _is_in_gate_corridor(self, pos: np.ndarray, gate_idx: int) -> bool:
        """Check if position is in gate passage corridor."""
        if gate_idx not in self._detected_gates:
            return False
        
        gate_info = self._detected_gates[gate_idx]
        gate_pos = gate_info['pos']
        gate_normal = gate_info['normal']
        
        # Vector from gate to position
        rel_pos = pos - gate_pos
        
        # Project onto gate normal (forward/backward)
        forward_dist = np.dot(rel_pos, gate_normal)
        
        # Check if within corridor length
        if abs(forward_dist) > self._gate_corridor_length:
            return False
        
        # Get perpendicular distance (lateral offset)
        lateral = rel_pos - forward_dist * gate_normal
        lateral_dist = np.linalg.norm(lateral[:2])  # XY plane
        
        return lateral_dist < self._gate_corridor_width / 2

    def _predict_trajectory(self, current_pos: np.ndarray, current_vel: np.ndarray,
                           target_vel: np.ndarray) -> list:
        """Predict future trajectory over MPC horizon."""
        trajectory = [current_pos.copy()]
        pos = current_pos.copy()
        vel = current_vel.copy()
        
        for _ in range(self._mpc_horizon):
            # Simple acceleration towards target velocity
            acc = (target_vel - vel) * 2.0  # proportional control
            vel = vel + acc * self._dt
            pos = pos + vel * self._dt
            trajectory.append(pos.copy())
        
        return trajectory

    def _compute_trajectory_cost(self, trajectory: list, des_pos: np.ndarray,
                                obstacles: list, in_gate_corridor: bool) -> float:
        """Compute cost for a predicted trajectory."""
        cost = 0.0
        
        # Tracking cost
        final_pos = trajectory[-1]
        tracking_error = np.linalg.norm(final_pos - des_pos)
        cost += 10.0 * tracking_error
        
        # Obstacle costs
        for pos in trajectory:
            for obs_info in obstacles:
                obs_pos = obs_info['pos']
                dist = np.linalg.norm(pos - obs_pos)
                
                if dist < self._obstacle_danger_radius:
                    # Very high penalty for collision
                    cost += 1000.0 * (self._obstacle_danger_radius - dist)
                elif dist < self._obstacle_influence_radius:
                    # Moderate penalty for being close
                    cost += 50.0 * (self._obstacle_influence_radius - dist)
        
        # If in gate corridor, penalize lateral deviation heavily
        if in_gate_corridor:
            for pos in trajectory:
                # Small lateral movements only
                lateral_movement = np.linalg.norm(pos[:2] - trajectory[0][:2])
                if lateral_movement > 0.1:
                    cost += 100.0 * lateral_movement
        
        return cost

    def _mpc_avoidance_offset(self, current_pos: np.ndarray, current_vel: np.ndarray,
                              des_pos: np.ndarray, des_vel: np.ndarray,
                              obstacles: list) -> np.ndarray:
        """Compute optimal position offset using MPC."""
        # Check if in gate corridor
        in_gate_corridor = False
        for gate_idx in self._detected_gates:
            if self._is_in_gate_corridor(current_pos, gate_idx):
                in_gate_corridor = True
                self._active_gate_idx = gate_idx
                break
        
        # If in gate corridor, don't deviate in XY
        if in_gate_corridor:
            return des_pos
        
        # Sample different velocity directions
        best_offset = np.zeros(2)
        best_cost = float('inf')
        
        # Baseline: no offset
        baseline_vel = des_vel.copy()
        baseline_traj = self._predict_trajectory(current_pos, current_vel, baseline_vel)
        baseline_cost = self._compute_trajectory_cost(baseline_traj, des_pos, obstacles, False)
        
        # Only search if baseline has obstacles
        if baseline_cost > 100.0:  # has obstacle penalty
            angles = np.linspace(0, 2*np.pi, self._mpc_sample_directions, endpoint=False)
            
            for angle in angles:
                for dist in self._mpc_sample_distances:
                    # Offset in XY plane only
                    offset_xy = dist * np.array([np.cos(angle), np.sin(angle)])
                    offset_3d = np.array([offset_xy[0], offset_xy[1], 0.0])
                    
                    # Target velocity towards offset position
                    target_pos = des_pos + offset_3d
                    target_vel = (target_pos - current_pos) * 2.0  # proportional
                    
                    # Predict trajectory
                    traj = self._predict_trajectory(current_pos, current_vel, target_vel)
                    
                    # Compute cost
                    cost = self._compute_trajectory_cost(traj, des_pos, obstacles, False)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_offset = offset_xy
        
        # Apply offset (XY only, preserve Z)
        adjusted_pos = des_pos.copy()
        adjusted_pos[:2] += best_offset
        
        return adjusted_pos

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control with MPC-based obstacle avoidance."""
        t = min(self._tick / self._freq, self._t_total)
        
        if t >= self._t_total:
            self._finished = True
        
        # Smooth waypoint updates
        trajectory_updated = False
        for wp_idx, target_pos in list(self._waypoint_targets.items()):
            current = self._current_waypoints[wp_idx]
            diff = target_pos - current
            
            if np.linalg.norm(diff) < 0.01:
                self._current_waypoints[wp_idx] = target_pos
                del self._waypoint_targets[wp_idx]
                trajectory_updated = True
            else:
                self._current_waypoints[wp_idx] += self._waypoint_update_rate * diff
                trajectory_updated = True
        
        if trajectory_updated:
            self._update_trajectory()
        
        # Get reference trajectory
        des_pos_ref = self._des_pos_spline(t)
        des_vel_ref = self._des_vel_spline(t)
        des_acc_ref = self._des_acc_spline(t)
        
        # MPC obstacle avoidance (XY only)
        obstacles_list = list(self._detected_obstacles.values())
        
        if len(obstacles_list) > 0 and not self._finished:
            des_pos = self._mpc_avoidance_offset(
                obs['pos'], obs['vel'],
                des_pos_ref, des_vel_ref,
                obstacles_list
            )
            
            # Recompute velocity towards adjusted position
            pos_error = des_pos - obs['pos']
            des_vel = des_vel_ref + 1.0 * pos_error
        else:
            des_pos = des_pos_ref
            des_vel = des_vel_ref
        
        # State command
        action = np.array([
            des_pos[0], des_pos[1], des_pos[2],
            des_vel[0], des_vel[1], des_vel[2],
            des_acc_ref[0], des_acc_ref[1], des_acc_ref[2],
            0.0, 0.0, 0.0, 0.0  # yaw, rates
        ], dtype=np.float32)
        
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
        """Update detections and waypoints."""
        self._tick += 1
        current_pos = obs['pos']
        
        # Detect gates
        if 'gates_pos' in obs and 'gates_quat' in obs:
            for i, gate_pos in enumerate(obs['gates_pos']):
                if i not in self._detected_gates:
                    if np.linalg.norm(gate_pos - current_pos) <= self._sensor_range:
                        rot = R.from_quat(obs['gates_quat'][i])
                        gate_normal = rot.as_matrix()[:, 0]
                        
                        self._detected_gates[i] = {
                            'pos': gate_pos.copy(),
                            'normal': gate_normal.copy()
                        }
                        
                        print(f"\n[GATE {i} DETECTED at tick {self._tick}]")
                        print(f"  Position: {gate_pos}")
                        
                        # Update waypoints for this gate
                        self._update_gate_waypoints(i, gate_pos, gate_normal)
        
        # Detect obstacles
        if 'obstacles_pos' in obs:
            for i, obs_pos in enumerate(obs['obstacles_pos']):
                if i not in self._detected_obstacles:
                    if np.linalg.norm(obs_pos - current_pos) <= self._sensor_range:
                        self._detected_obstacles[i] = {
                            'pos': obs_pos.copy()
                        }
                        print(f"\n[OBSTACLE {i} DETECTED at tick {self._tick}]")
        
        return self._finished

    def _update_gate_waypoints(self, gate_idx: int, gate_pos: np.ndarray, 
                               gate_normal: np.ndarray):
        """Update waypoints for a detected gate."""
        # Find waypoints corresponding to this gate
        # Assume structure: [start, approach_0, center_0, exit_0, approach_1, ...]
        base_idx = 1 + gate_idx * 3
        
        if base_idx + 2 < len(self._current_waypoints):
            # Approach waypoint
            approach_target = gate_pos - 0.8 * gate_normal
            approach_target[2] = gate_pos[2]
            self._waypoint_targets[base_idx] = approach_target
            
            # Center waypoint
            self._waypoint_targets[base_idx + 1] = gate_pos.copy()
            
            # Exit waypoint
            exit_target = gate_pos + 0.6 * gate_normal
            exit_target[2] = gate_pos[2]
            self._waypoint_targets[base_idx + 2] = exit_target
            
            print(f"  Updated waypoints {base_idx}, {base_idx+1}, {base_idx+2}")

    def episode_callback(self):
        """Reset for new episode."""
        self.i_error[:] = 0
        self._tick = 0
        self._finished = False
        self._detected_gates.clear()
        self._detected_obstacles.clear()
        self._waypoint_targets.clear()
        self._active_gate_idx = None
        self._current_waypoints = self._original_waypoints.copy()
        self._update_trajectory()
        print("\n[EPISODE RESET] MPC Controller reset\n")