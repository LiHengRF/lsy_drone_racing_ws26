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
        self._mpc_horizon = 15
        self._mpc_sample_directions = 16
        self._mpc_sample_distances = [0.05, 0.10, 0.15, 0.20]
        
        # Obstacle avoidance parameters - XY plane distances
        self._obstacle_influence_radius = 0.6
        self._obstacle_danger_radius = 0.3  # obstacle(10cm) + drone(10cm)
        self._gate_corridor_width = 0.4
        self._gate_corridor_length = 0.6  # Reduced from 1.5m to avoid overlap
        
        # Tracking parameters
        self._detected_gates = {}
        self._detected_obstacles = {}
        self._active_gate_idx = None
        
        # Waypoint updates
        self._waypoint_update_rate = 0.7
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
        
        # For sequential waypoint updates
        self._last_obstacle_count = 0
        self._last_updated_gates = set()  # Track which gates have been updated
        
        print("="*60)
        print(f"MPC horizon: {self._mpc_horizon} steps ({self._mpc_horizon * self._dt:.2f}s)")
        print(f"Sample directions: {self._mpc_sample_directions}")
        print(f"Initial waypoints: {len(self._original_waypoints)}")
        print("="*60 + "\n")

    def _check_waypoint_safety(self, pos: np.ndarray, obstacles: list, 
                              gate_pos: np.ndarray = None) -> tuple[bool, float]:
        """Check if a waypoint position is safe (>20cm from obstacles).
        
        If gate_pos is provided, also check the path from gate to waypoint.
        """
        min_dist = float('inf')
        
        for obs_info in obstacles:
            obs_pos = obs_info['pos']
            
            # Check point distance
            dist = np.linalg.norm(pos[:2] - obs_pos[:2])
            min_dist = min(min_dist, dist)
            
            # If gate position provided, check path distance
            if gate_pos is not None:
                seg_vec = pos[:2] - gate_pos[:2]
                seg_length = np.linalg.norm(seg_vec)
                
                if seg_length > 1e-6:
                    seg_dir = seg_vec / seg_length
                    to_obs = obs_pos[:2] - gate_pos[:2]
                    proj_length = np.dot(to_obs, seg_dir)
                    proj_length = np.clip(proj_length, 0, seg_length)
                    
                    closest_point = gate_pos[:2] + proj_length * seg_dir
                    path_dist = np.linalg.norm(closest_point - obs_pos[:2])
                    min_dist = min(min_dist, path_dist)
        
        is_safe = min_dist >= 0.25
        return is_safe, min_dist

    def _generate_safe_approach_exit(self, gate_pos: np.ndarray, gate_normal: np.ndarray, 
                                     distance: float, obstacles: list) -> np.ndarray:
        """Generate safe approach/exit point, avoiding obstacles if needed."""
        # Try default position
        default_pos = gate_pos + distance * gate_normal
        default_pos[2] = gate_pos[2]
        
        # Check both point safety and path safety
        is_safe, min_dist = self._check_waypoint_safety(default_pos, obstacles, gate_pos)
        if is_safe:
            return default_pos
        
        # Try lateral offsets
        perpendicular = np.array([-gate_normal[1], gate_normal[0], 0.0])
        if np.linalg.norm(perpendicular) > 1e-6:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        best_pos = default_pos
        best_dist = min_dist
        
        for sign in [1, -1]:
            for offset in [0.3, 0.4, 0.5, 0.6, 0.7]:  # Extended to 0.7m
                test_pos = gate_pos + distance * gate_normal + sign * offset * perpendicular
                test_pos[2] = gate_pos[2]
                
                is_safe, test_dist = self._check_waypoint_safety(test_pos, obstacles, gate_pos)
                if is_safe:
                    return test_pos
                
                # Track best even if not fully safe
                if test_dist > best_dist:
                    best_dist = test_dist
                    best_pos = test_pos
        
        # Return best effort with warning
        return best_pos

    def _generate_waypoints(self) -> np.ndarray:
        """Generate initial waypoints."""
        waypoints = []
        waypoints.append(np.array([-1.5, 0.75, 0.01])) # Start
        
        for i, gate in enumerate(self._nominal_gates):
            gate_pos = gate['pos']
            gate_rpy = gate['rpy']
            rot = R.from_euler('xyz', gate_rpy)
            gate_normal = rot.as_matrix()[:, 0]
            
            # Safe approach point
            approach = self._generate_safe_approach_exit(
                gate_pos, -gate_normal, 0.8, self._nominal_obstacles
            )
            waypoints.append(approach)
            
            # Center
            waypoints.append(gate_pos.copy())
            
            # Safe exit point
            exit_pt = self._generate_safe_approach_exit(
                gate_pos, gate_normal, 0.8, self._nominal_obstacles
            )
            waypoints.append(exit_pt)
            
            # SPECIAL: Add intermediate point after Gate2 to avoid obstacle C during climb
            if i == 2:  # After Gate 2 (0-indexed)
                # Gate2 is at z=0.7, Gate3 approach will be higher
                # Add intermediate point that is safe from obstacle C
                intermediate = exit_pt.copy()
                intermediate[2] = 0.7  # Keep low initially
                
                # Move further away from obstacle direction
                # Gate2 faces west (-1,0,0), obstacle C is west of gate
                # So move intermediate point in perpendicular direction (north/south)
                if exit_pt[1] > gate_pos[1]:  # Already offset north
                    intermediate[1] += 0.3  # Move further north
                else:  # Offset south
                    intermediate[1] -= 0.3  # Move further south
                    
                waypoints.append(intermediate)
        
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
            acc = (target_vel - vel) * 1.5
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
        
        # Obstacle costs using XY distance
        for pos in trajectory:
            for obs_info in obstacles:
                obs_pos = obs_info['pos']
                
                # Use XY plane distance (obstacle is a vertical pole, not a 3D point)
                dist = np.linalg.norm(pos[:2] - obs_pos[:2])
                
                # Check if Z is in obstacle height range
                if 0.0 <= pos[2] <= 1.6:
                    if dist < self._obstacle_danger_radius:
                        # Very high penalty for collision
                        cost += 1000.0 * (self._obstacle_danger_radius - dist)
                    elif dist < self._obstacle_influence_radius:
                        # Moderate penalty for being close
                        cost += 50.0 * (self._obstacle_influence_radius - dist)
        
        # ADDED: Penalty for getting close to already-passed gates (avoid backward collision)
        for gate_idx, gate_info in self._detected_gates.items():
            gate_pos = gate_info['pos']
            
            # Check if this is a recently passed gate (not the current target)
            if gate_idx != self._active_gate_idx:
                for pos in trajectory:
                    # Check XY distance to gate center
                    gate_dist = np.linalg.norm(pos[:2] - gate_pos[:2])
                    
                    # High penalty if getting close to a non-active gate
                    if gate_dist < 0.5:  # 50cm danger zone around gates
                        cost += 500.0 * (0.5 - gate_dist)
        
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
        
        # Determine forward direction to avoid backward collision
        forward_direction = des_pos[:2] - current_pos[:2]
        forward_norm = np.linalg.norm(forward_direction)
        if forward_norm > 1e-3:
            forward_direction = forward_direction / forward_norm
        else:
            # Use velocity direction
            forward_direction = des_vel[:2] / (np.linalg.norm(des_vel[:2]) + 1e-6)
        
        # Sample different velocity directions
        best_offset = np.zeros(2)
        best_cost = float('inf')
        
        # Baseline: no offset
        baseline_vel = des_vel.copy()
        baseline_traj = self._predict_trajectory(current_pos, current_vel, baseline_vel)
        baseline_cost = self._compute_trajectory_cost(baseline_traj, des_pos, obstacles, False)
        
        # Only search if baseline has obstacles
        if baseline_cost > 100.0:
            # Limit sampling to forward hemisphere to avoid backward collision
            angles = np.linspace(-np.pi*0.7, np.pi*0.7, self._mpc_sample_directions, endpoint=True)
            
            for angle in angles:
                for dist in self._mpc_sample_distances:
                    # Offset in XY plane only
                    offset_xy = dist * np.array([np.cos(angle), np.sin(angle)])
                    
                    # Rotate offset to align with forward direction
                    forward_angle = np.arctan2(forward_direction[1], forward_direction[0])
                    cos_f = np.cos(forward_angle)
                    sin_f = np.sin(forward_angle)
                    rotated_offset = np.array([
                        offset_xy[0] * cos_f - offset_xy[1] * sin_f,
                        offset_xy[0] * sin_f + offset_xy[1] * cos_f
                    ])
                    
                    offset_3d = np.array([rotated_offset[0], rotated_offset[1], 0.0])
                    
                    # Target velocity towards offset position
                    target_pos = des_pos + offset_3d
                    target_vel = (target_pos - current_pos) * 2.0
                    
                    # Predict trajectory
                    traj = self._predict_trajectory(current_pos, current_vel, target_vel)
                    
                    # Compute cost
                    cost = self._compute_trajectory_cost(traj, des_pos, obstacles, False)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_offset = rotated_offset
        
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
        """Update detections and waypoints sequentially."""
        self._tick += 1
        current_pos = obs['pos']
        
        # Track newly detected gates (don't update immediately)
        newly_detected_gates = []
        
        # Detect gates
        if 'gates_pos' in obs and 'gates_quat' in obs:
            for i, gate_pos in enumerate(obs['gates_pos']):
                if i not in self._detected_gates:
                    if np.linalg.norm(gate_pos[:2] - current_pos[:2]) <= self._sensor_range:
                        rot = R.from_quat(obs['gates_quat'][i])
                        gate_normal = rot.as_matrix()[:, 0]
                        
                        self._detected_gates[i] = {
                            'pos': gate_pos.copy(),
                            'normal': gate_normal.copy()
                        }
                        newly_detected_gates.append(i)
                        
                        print(f"\n[GATE {i} DETECTED at tick {self._tick}]")
                        print(f"  Position: {gate_pos}")
        
        # Detect obstacles
        newly_detected_obstacles = False
        if 'obstacles_pos' in obs:
            for i, obs_pos in enumerate(obs['obstacles_pos']):
                if i not in self._detected_obstacles:
                    if np.linalg.norm(obs_pos[:2] - current_pos[:2]) <= self._sensor_range:
                        self._detected_obstacles[i] = {
                            'pos': obs_pos.copy()
                        }
                        newly_detected_obstacles = True
                        print(f"\n[OBSTACLE {i} DETECTED at tick {self._tick}]")
        
        # Sequential waypoint update logic
        # Update waypoints in order, only when all previous gates are detected
        if newly_detected_gates or newly_detected_obstacles:
            detected_gate_indices = sorted(self._detected_gates.keys())
            
            for gate_idx in detected_gate_indices:
                # Check if all previous gates are detected
                all_previous_detected = all(
                    prev_idx in self._detected_gates 
                    for prev_idx in range(gate_idx)
                )
                
                # Update if: (1) it's the first gate, OR (2) all previous gates detected
                # AND either: (a) gate just detected, OR (b) new obstacle detected
                should_update = (gate_idx == 0 or all_previous_detected) and (
                    gate_idx in newly_detected_gates or 
                    (newly_detected_obstacles and gate_idx in self._last_updated_gates)
                )
                
                if should_update:
                    gate_info = self._detected_gates[gate_idx]
                    self._update_gate_waypoints(
                        gate_idx, 
                        gate_info['pos'], 
                        gate_info['normal']
                    )
                    self._last_updated_gates.add(gate_idx)
                    print(f"  → Updated waypoints for Gate {gate_idx} (sequential order)")
        
        self._last_obstacle_count = len(self._detected_obstacles)
        return self._finished

    def _update_gate_waypoints(self, gate_idx: int, gate_pos: np.ndarray, 
                               gate_normal: np.ndarray):
        """Update waypoints for a detected gate."""
        base_idx = 1 + gate_idx * 3
        
        if base_idx + 2 < len(self._current_waypoints):
            detected_obstacles = list(self._detected_obstacles.values())
            
            # Approach waypoint
            approach_target = self._generate_safe_approach_exit(
                gate_pos, -gate_normal, 0.7, detected_obstacles
            )
            self._waypoint_targets[base_idx] = approach_target
            
            # Center waypoint
            self._waypoint_targets[base_idx + 1] = gate_pos.copy()
            
            # Exit waypoint (increased distance for safety)
            exit_target = self._generate_safe_approach_exit(
                gate_pos, gate_normal, 0.7, detected_obstacles
            )
            self._waypoint_targets[base_idx + 2] = exit_target
            
            print(f"Updated waypoints {base_idx}, {base_idx+1}, {base_idx+2}")

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
        
        # Reset sequential update tracking
        self._last_obstacle_count = 0
        self._last_updated_gates.clear()