"""
StateController
Author: Yuming Li
Date: 29.10.2025

Description:
  This controller uses cubic spline interpolation to generate smooth trajectories through waypoints.
It supports dynamic replanning when gates or obstacles change position during flight.
  This controller is a refactored and extended version of the EasyController 
from the LSY Drone Racing project by Yufei Hua (Learning Systems and Robotics Lab, TUM).
It is used solely for learning and research purposes.

Key features:
- Pre-computed trajectory with cubic spline interpolation
- Collision avoidance with obstacles
- Dynamic replanning on environment changes
- Real-time 3D visualization
- Adding detour waypoints for backtracking gates

Original repository:
https://github.com/yufei4hua/lsy_drone_racing

License:
  This file is derived from code released under the MIT License.
  Copyright (c) 2024 Learning Systems and Robotics Lab (LSY)
  See the original license at the above repository for details.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.utils import draw_line

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """Trajectory-following controller for drone racing.
    
    This controller plans a smooth trajectory through predefined waypoints and tracks it
    over time. It can dynamically replan when the environment changes.
    """

    # Class constants
    TRAJECTORY_DURATION = 15.0  # Total trajectory duration in seconds
    STATE_DIMENSION = 13  # [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
    OBSTACLE_SAFETY_DISTANCE = 0.3  # Minimum distance to obstacles in meters
    VISUALIZATION_SAMPLES = 100  # Number of points for trajectory visualization
    LOG_INTERVAL = 100  # Print debug info every N ticks

    def __init__(
        self, 
        obs: dict[str, NDArray[np.floating]], 
        info: dict, 
        config: dict
    ):
        """Initialize the controller.

        Args:
            obs: Initial observation containing drone state, gates, and obstacles.
            info: Initial environment information from reset.
            config: Race configuration with environment frequency and settings.
        """
        super().__init__(obs, info, config)
        
        # Controller state
        self._time_step = 0
        self._control_frequency = config.env.freq
        self._is_finished = False
        
        # Environment state tracking for change detection
        self._last_gate_flags = None
        self._last_obstacle_flags = None

        # === DEBUG: Initialize debug attributes ===
        self._debug_detour_analysis = []  # Will store analysis for each gate pair
        self._debug_detour_summary = {}   # Will store overall summary
        self._debug_detour_waypoints_added = []
        self._debug_waypoints_initial = None
        self._debug_waypoints_after_detour = None
        self._debug_waypoints_final = None

        # Extract gate information
        self.gate_positions = obs['gates_pos']
        self.gate_normals, self.gate_y_axes, self.gate_z_axes = \
        self._extract_gate_coordinate_frames(obs['gates_quat'])
        
        # Extract obstacle information
        self.obstacle_positions = obs['obstacles_pos']
        
        # Initial drone position
        self.initial_position = obs['pos']
        
        # Enable visualization (trajectory plotting)
        self.visualization = False

        # Calculate waypoints 
        waypoints = self.calc_waypoints_from_gates(
            self.initial_position,
            self.gate_positions,
            self.gate_normals,
            approach_distance=0.5,
            num_intermediate_points=5
        )
        print(f"Initial waypoints count: {len(waypoints)}")
        print(f"Initial waypoints:\n{waypoints}")
                
        # === DEBUG: Save initial waypoints ===
        self._debug_waypoints_initial = waypoints.copy()
        
        # Step 2: Add detour waypoints for backtracking gates (NEW!)
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes,
            num_intermediate_points=5,
            angle_threshold=120.0,
            detour_distance=0.65
        )
        print(f"Waypoints after detour: {len(waypoints)}")

        # === DEBUG: Save waypoints after detour ===
        self._debug_waypoints_after_detour = waypoints.copy()
        
        # Apply collision avoidance
        time_params, waypoints = self._avoid_collisions(
            waypoints, 
            self.obstacle_positions,
            self.OBSTACLE_SAFETY_DISTANCE
        )

        # === DEBUG: Save final waypoints ===
        self._debug_waypoints_final = waypoints.copy()
        
        # Generate smooth trajectory
        self.trajectory = self._generate_trajectory(self.TRAJECTORY_DURATION, waypoints)
        
        # Initialize visualization
        self.fig = None
        self.ax = None
        if self.visualization:
            self._visualize_trajectory(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=self.obstacle_positions,
                trajectory=self.trajectory,
                waypoints=waypoints,
                drone_position=obs['pos']
            )

        print("=== Available info keys ===")
        print(info.keys())
        print("\n=== Available obs keys ===")
        print(obs.keys())


    def _extract_gate_normals(self, gates_quaternions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Extract gate normal vectors from quaternions.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [w, x, y, z].
            
        Returns:
            Array of gate normal vectors (first column of rotation matrices).
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        return rotation_matrices[:, :, 0]  # Extract first column (x-axis / normal)

    def _extract_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Extract complete local coordinate frames for all gates.
        
        Args:
            gates_quaternions: Array of gate orientations as quaternions [w, x, y, z].
            
        Returns:
            Tuple of (normals, y_axes, z_axes) where:
            - normals: Gate normal vectors (x-axis, penetration direction)
            - y_axes: Gate width directions (left-right)
            - z_axes: Gate height directions (up-down)
        """
        rotations = Rotation.from_quat(gates_quaternions)
        rotation_matrices = rotations.as_matrix()
        
        normals = rotation_matrices[:, :, 0]  # x-axis (penetration direction)
        y_axes = rotation_matrices[:, :, 1]   # y-axis (width direction)
        z_axes = rotation_matrices[:, :, 2]   # z-axis (height direction)
        
        return normals, y_axes, z_axes

    def calc_waypoints_from_gates(
        self,
        initial_position: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        approach_distance: float = 0.5,
        num_intermediate_points: int = 5
    ) -> NDArray[np.floating]:
        """Automatically generate waypoints based on gate positions.
        
        Creates multiple waypoints around each gate to ensure smooth passage.
        
        Args:
            initial_position: Starting position of the drone.
            gate_positions: Array of gate center positions.
            gate_normals: Array of gate normal vectors (penetration direction).
            approach_distance: Distance before/after gate center for waypoints.
            num_intermediate_points: Number of waypoints per gate.
            
        Returns:
            Array of waypoints including initial position.
        """
        num_gates = len(gate_positions)
        waypoints_per_gate = []
        
        for i in range(num_intermediate_points):
            # Interpolate from -approach_distance to +approach_distance
            offset = -approach_distance + (i / (num_intermediate_points - 1)) * 2 * approach_distance
            
            # Create waypoints for all gates at this offset
            gate_waypoints = gate_positions + offset * gate_normals
            waypoints_per_gate.append(gate_waypoints)
        
        # Reshape to (num_gates * num_intermediate_points, 3)
        waypoints = np.concatenate(waypoints_per_gate, axis=1)
        waypoints = waypoints.reshape(num_gates, num_intermediate_points, 3).reshape(-1, 3)
        
        # Prepend initial position
        waypoints = np.vstack([initial_position, waypoints])
        
        return waypoints

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        """Add detour waypoints for gates that require backtracking.
        
        When two consecutive gates face each other (backtracking scenario),
        this method inserts a detour waypoint to create a smooth arc around the gate.
        
        Args:
            waypoints: Original waypoints array with shape (N, 3).
            gate_positions: Array of gate center positions.
            gate_normals: Array of gate normal vectors.
            gate_y_axes: Array of gate y-axis vectors (width direction).
            gate_z_axes: Array of gate z-axis vectors (height direction).
            num_intermediate_points: Number of waypoints per gate (used for indexing).
            angle_threshold: Angle threshold (degrees) to detect backtracking.
            detour_distance: Distance to offset the detour waypoint.
            
        Returns:
            Modified waypoints array with detour waypoints inserted.
        """
        waypoints_list = list(waypoints)
        inserted_count = 0
        num_gates = len(gate_positions)
        
        print("\n=== Detour Waypoint Analysis ===")
        
        # Check each pair of consecutive gates
        for i in range(num_gates - 1):
            # === DEBUG: Initialize debug info for this gate pair ===
            debug_info = {
                'gate_pair': f"Gate {i} -> Gate {i+1}",
                'gate_i_idx': i,
                'gate_i_plus_1_idx': i + 1,
            }
            
            print(f"\nGate {i} -> Gate {i+1}:")
            
            # Calculate indices accounting for previously inserted waypoints
            # last_idx_gate_i: index of the last waypoint of gate i
            # first_idx_gate_i_plus_1: index of the first waypoint of gate i+1
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            # === DEBUG: Store waypoint indices ===
            debug_info['last_idx_gate_i'] = last_idx_gate_i
            debug_info['first_idx_gate_i_plus_1'] = first_idx_gate_i_plus_1
            
            # Get the two waypoints
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            
            # === DEBUG: Store waypoint coordinates ===
            debug_info['p1_coords'] = p1.copy()
            debug_info['p2_coords'] = p2.copy()
            
            # Calculate vector from p1 to p2
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            
            # === DEBUG: Store vector info ===
            debug_info['v_vector'] = v.copy()
            debug_info['v_norm'] = v_norm
            
            if v_norm < 1e-6:
                print(f"  Vector too short (norm={v_norm:.6f}), skipping")
                debug_info['skipped'] = True
                debug_info['skip_reason'] = 'vector_too_short'
                self._debug_detour_analysis.append(debug_info)
                continue
            
            # Get gate i's normal vector and local coordinate frame
            normal_i = gate_normals[i]
            y_axis = gate_y_axes[i]
            z_axis = gate_z_axes[i]
            gate_center = gate_positions[i]
            
            # === DEBUG: Store gate info ===
            debug_info['gate_center'] = gate_center.copy()
            debug_info['normal_i'] = normal_i.copy()
            debug_info['y_axis'] = y_axis.copy()
            debug_info['z_axis'] = z_axis.copy()
            
            # Calculate angle between gate normal and trajectory direction
            cos_angle = np.dot(normal_i, v) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            # === DEBUG: Store angle info ===
            debug_info['cos_angle'] = cos_angle
            debug_info['angle_degrees'] = angle_deg
            
            print(f"  Angle between normal and trajectory: {angle_deg:.1f}°")
            
            # Check if backtracking is needed (angle > threshold means going backwards)
            if angle_deg > angle_threshold:
                print(f"  ⚠ Backtracking detected (angle={angle_deg:.1f}° > threshold={angle_threshold}°)")
                
                # === DEBUG: Mark as needs detour ===
                debug_info['needs_detour'] = True
                
                # Project trajectory vector onto gate plane (perpendicular to normal)
                # This gives us the direction to move in the gate's local frame
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                # === DEBUG: Store projection info ===
                debug_info['v_proj'] = v_proj.copy()
                debug_info['v_proj_norm'] = v_proj_norm
                
                # Determine detour direction based on projected vector
                if v_proj_norm < 1e-6:
                    # If projection is negligible, default to right side
                    detour_direction_vector = y_axis
                    detour_direction_name = 'default_right (+y_axis)'
                    proj_angle_deg = 0.0
                else:
                    # Step 2: Calculate components in local coordinate system
                    v_proj_y = np.dot(v_proj, y_axis)  # Left-right component
                    v_proj_z = np.dot(v_proj, z_axis)  # Up-down component
                    
                    # === DEBUG: Store local components ===
                    debug_info['v_proj_y_component'] = v_proj_y
                    debug_info['v_proj_z_component'] = v_proj_z
                    
                    # Step 3: Calculate angle in gate plane
                    # angle = 0° means +y direction (right)
                    # angle = 90° means +z direction (up)
                    # angle = ±180° means -y direction (left)
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    # === DEBUG: Store angle ===
                    debug_info['projection_angle_degrees'] = proj_angle_deg
                    
                    # Step 4: Determine detour direction based on angle
                    if -90 <= proj_angle_deg < 45:
                        # Right side
                        detour_direction_vector = y_axis
                        detour_direction_name = 'right (+y_axis)'
                    elif 45 <= proj_angle_deg < 135:
                        # Top side
                        detour_direction_vector = z_axis
                        detour_direction_name = 'top (+z_axis)'
                    else:  # angle >= 135 or angle < -90
                        # Left side
                        detour_direction_vector = -y_axis
                        detour_direction_name = 'left (-y_axis)'
                    
                    print(f"  Projection angle: {proj_angle_deg:.1f}° → Detour direction: {detour_direction_name}")
                
                # === DEBUG: Store direction choice ===
                debug_info['detour_direction_vector'] = detour_direction_vector.copy()
                debug_info['detour_direction_name'] = detour_direction_name
                debug_info['projection_angle_degrees'] = proj_angle_deg
                
                # Step 5: Calculate detour waypoint
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                
                # === DEBUG: Store detour waypoint ===
                debug_info['detour_waypoint'] = detour_waypoint.copy()
                debug_info['detour_direction'] = detour_direction_name
                
                # Also add to the separate tracking list
                self._debug_detour_waypoints_added.append({
                    'gate_index': i,
                    'waypoint_coords': detour_waypoint.copy(),
                    'direction': detour_direction_name
                })
                
                # Step 6: Insert the detour waypoint into the waypoints list  
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
    
                # === DEBUG: Store insertion info ===
                debug_info['insert_position'] = insert_position
                debug_info['inserted'] = True
                
                print(f"  Inserted detour waypoint at index {insert_position}")
                print(f"  Detour coords: [{detour_waypoint[0]:.3f}, {detour_waypoint[1]:.3f}, {detour_waypoint[2]:.3f}]")
            else:
                print(f"  ✓ No backtracking detected, proceeding normally")
                debug_info['needs_detour'] = False
                debug_info['inserted'] = False
            
            # === DEBUG: Store current inserted count ===
            debug_info['total_inserted_so_far'] = inserted_count
            
            # Add this gate pair's debug info to the list
            self._debug_detour_analysis.append(debug_info)
        
        # === DEBUG: Store final summary ===
        self._debug_detour_summary = {
            'total_detours_added': inserted_count,
            'original_waypoint_count': len(waypoints),
            'final_waypoint_count': len(waypoints_list),
            'num_gate_pairs_checked': num_gates - 1,
            'detour_waypoints': self._debug_detour_waypoints_added  # Also include in summary
        }
        
        print(f"\n=== Total detour waypoints added: {inserted_count} ===")
        
        # === DEBUG: Print all added detour waypoints for easy viewing ===
        if self._debug_detour_waypoints_added:
            print("\n=== Added Detour Waypoints ===")
            for idx, detour_info in enumerate(self._debug_detour_waypoints_added):
                coords = detour_info['waypoint_coords']
                gate_idx = detour_info['gate_index']
                print(f"  Detour #{idx+1} (Gate {gate_idx}): "
                    f"[{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}] - {detour_info['direction']}")
        print()
        
        return np.array(waypoints_list)

    def _avoid_collisions(
        self,
        waypoints: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating],
        safety_distance: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Adjust waypoint timing to avoid obstacles.
        
        Slows down near obstacles by adjusting time parameters for the spline.
        
        Args:
            waypoints: Array of waypoints with shape (N, 3).
            obstacle_positions: Array of obstacle positions.
            safety_distance: Minimum safe distance from obstacles.
            
        Returns:
            Tuple of (time_parameters, waypoints) where time_parameters define
            the timing for each waypoint in the spline.
        """
        num_waypoints = len(waypoints)
        time_params = np.zeros(num_waypoints)
        
        for i in range(num_waypoints):
            # Calculate distance to nearest obstacle
            if len(obstacle_positions) > 0:
                distances = np.linalg.norm(
                    obstacle_positions - waypoints[i],
                    axis=1
                )
                min_distance = np.min(distances)
            else:
                min_distance = float('inf')
            
            # Slow down near obstacles
            if min_distance < safety_distance:
                time_factor = 2.0  # Double the time near obstacles
            else:
                time_factor = 1.0
            
            # Accumulate time
            if i == 0:
                time_params[i] = 0.0
            else:
                segment_length = np.linalg.norm(waypoints[i] - waypoints[i-1])
                time_params[i] = time_params[i-1] + segment_length * time_factor
        
        # Normalize time parameters to [0, 1]
        if time_params[-1] > 0:
            time_params = time_params / time_params[-1]
        
        return time_params, waypoints

    def _generate_trajectory(
        self,
        total_duration: float,
        waypoints: NDArray[np.floating]
    ) -> CubicSpline:
        """Generate smooth trajectory through waypoints using cubic spline.
        
        Args:
            total_duration: Total time duration for the trajectory.
            waypoints: Array of waypoints to interpolate through.
            
        Returns:
            CubicSpline object that can be sampled at any time t in [0, total_duration].
        """
        num_waypoints = len(waypoints)
        
        # Collision-aware time parameterization
        time_params, waypoints = self._avoid_collisions(
            waypoints,
            self.obstacle_positions,
            self.OBSTACLE_SAFETY_DISTANCE
        )
        
        # Scale to total duration
        time_points = time_params * total_duration
        
        # Create cubic spline
        trajectory = CubicSpline(
            time_points,
            waypoints,
            bc_type='clamped'  # Zero derivatives at endpoints
        )
        
        return trajectory

    def _detect_environment_change(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Detect if gates or obstacles have changed position.
        
        Args:
            obs: Current observation.
            
        Returns:
            True if environment has changed, False otherwise.
        """
        current_gate_flags = obs.get('gates_in_range', None)
        current_obstacle_flags = obs.get('obstacles_in_range', None)
        
        # Check for changes
        gates_changed = (
            self._last_gate_flags is not None and
            current_gate_flags is not None and
            not np.array_equal(self._last_gate_flags, current_gate_flags)
        )
        
        obstacles_changed = (
            self._last_obstacle_flags is not None and
            current_obstacle_flags is not None and
            not np.array_equal(self._last_obstacle_flags, current_obstacle_flags)
        )
        
        # Update stored flags
        self._last_gate_flags = current_gate_flags
        self._last_obstacle_flags = current_obstacle_flags
        
        return gates_changed or obstacles_changed

    def _replan_trajectory(self, obs: dict[str, NDArray[np.floating]], current_time: float):
        """Replan the trajectory when environment changes.
        
        Args:
            obs: Current observation with updated gate/obstacle positions.
            current_time: Current time along the trajectory.
        """
        print(f"\n=== Replanning trajectory at t={current_time:.2f}s ===")
        
        # Update gate and obstacle information
        self.gate_positions = obs['gates_pos']
        self.gate_normals = self._extract_gate_normals(obs['gates_quat'])
        self.obstacle_positions = obs['obstacles_pos']
        
        # Recalculate waypoints starting from current position
        current_position = obs['pos']
        waypoints = self.calc_waypoints_from_gates(
            current_position,
            self.gate_positions,
            self.gate_normals
        )
        
        # Add detour waypoints
        waypoints = self._add_detour_waypoints(
            waypoints,
            self.gate_positions,
            self.gate_normals,
            self.gate_y_axes,
            self.gate_z_axes
        )
        
        # Regenerate trajectory
        remaining_time = self.TRAJECTORY_DURATION - current_time
        self.trajectory = self._generate_trajectory(remaining_time, waypoints)
        
        # Reset time step to align with new trajectory
        self._time_step = 0
        
        print("=== Replanning complete ===\n")

    def _visualize_trajectory(
        self,
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        obstacle_positions: NDArray[np.floating] | None = None,
        trajectory: CubicSpline | None = None,
        waypoints: NDArray[np.floating] | None = None,
        drone_position: NDArray[np.floating] | None = None
    ):
        """Visualize the trajectory, gates, and obstacles in 3D.
        
        Args:
            gate_positions: Array of gate center positions.
            gate_normals: Array of gate normal vectors.
            obstacle_positions: Optional array of obstacle positions.
            trajectory: Optional trajectory spline to plot.
            waypoints: Optional waypoints to plot.
            drone_position: Optional current drone position.
        """
        if plt is None:
            print("Matplotlib not available, skipping visualization")
            return
        
        # Create figure if it doesn't exist
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()  # Interactive mode
        
        # Clear previous plot
        self.ax.clear()
        
        # Plot gates
        self.ax.scatter(
            gate_positions[:, 0],
            gate_positions[:, 1],
            gate_positions[:, 2],
            c='blue',
            s=100,
            marker='o',
            label='Gates'
        )
        
        # Plot gate normals
        for i, (pos, normal) in enumerate(zip(gate_positions, gate_normals)):
            self.ax.quiver(
                pos[0], pos[1], pos[2],
                normal[0], normal[1], normal[2],
                length=0.3,
                color='blue',
                alpha=0.6
            )
        
        # Plot obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            self.ax.scatter(
                obstacle_positions[:, 0],
                obstacle_positions[:, 1],
                obstacle_positions[:, 2],
                c='red',
                s=100,
                marker='x',
                label='Obstacles'
            )
        
        # Plot trajectory
        if trajectory is not None:
            t_samples = np.linspace(0, self.TRAJECTORY_DURATION, self.VISUALIZATION_SAMPLES)
            trajectory_points = trajectory(t_samples)
            self.ax.plot(
                trajectory_points[:, 0],
                trajectory_points[:, 1],
                trajectory_points[:, 2],
                'g-',
                linewidth=2,
                label='Trajectory'
            )
        
        # Plot waypoints
        if waypoints is not None:
            self.ax.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                waypoints[:, 2],
                c='orange',
                s=50,
                marker='^',
                label='Waypoints'
            )
        
        # Plot drone position
        if drone_position is not None:
            self.ax.scatter(
                drone_position[0],
                drone_position[1],
                drone_position[2],
                c='purple',
                s=150,
                marker='*',
                label='Drone'
            )
        
        # Set labels and title
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Drone Racing Trajectory')
        self.ax.legend()
        
        # Update plot
        plt.draw()
        plt.pause(0.001)
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state for the drone.
        
        Args:
            obs: Current observation of environment state.
            info: Optional additional information.
            
        Returns:
            13D state vector [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
            Only position (first 3 elements) is set; rest are zeros for low-level controller.
        """
        # Compute current time along trajectory
        current_time = min(self._time_step / self._control_frequency, self.TRAJECTORY_DURATION)
        
        # Sample target position from trajectory
        target_position = self.trajectory(current_time)
        
        # Periodic logging
        if self._time_step % self.LOG_INTERVAL == 0:
            print(f"Time: {current_time:.2f}s | "
                  f"Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # Check for environment changes and replan if necessary
        if self._detect_environment_change(obs):
            self._replan_trajectory(obs, current_time)
        if self.visualization:
            self._visualize_trajectory(
                self.gate_positions,
                self.gate_normals,
                obstacle_positions=obs['obstacles_pos'],
                trajectory=self.trajectory,
                drone_position=obs['pos']
            )
        # Check if trajectory is complete
        if current_time >= self.TRAJECTORY_DURATION:
            self._is_finished = True
        
        # Draw trajectory in simulation environment (if available)
        try:
            draw_line(self.env, self.trajectory(self.trajectory.x), 
                     rgba=np.array([1.0, 1.0, 1.0, 0.2]))
        except (AttributeError, TypeError):
            pass  # env not available or draw_line not supported
        
        # Return 13D state with only position filled
        return np.concatenate((target_position, np.zeros(10)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        """Called after each environment step.
        
        Args:
            action: Action taken.
            obs: Resulting observation.
            reward: Reward received.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Additional information.
            
        Returns:
            True if controller is finished, False otherwise.
        """
        self._time_step += 1
        return self._is_finished

    # ==================== Utility Methods for External Use ====================
    
    def get_trajectory_function(self) -> CubicSpline:
        """Get the trajectory spline function.
        
        Returns:
            CubicSpline object representing the trajectory.
        """
        return self.trajectory

    def get_trajectory_waypoints(self) -> NDArray[np.floating]:
        """Get discrete waypoints sampled from trajectory at control frequency.
        
        Returns:
            Array of waypoints with shape (num_timesteps, 3).
        """
        time_samples = np.linspace(0, self.TRAJECTORY_DURATION,
                                   int(self._control_frequency * self.TRAJECTORY_DURATION))
        return self.trajectory(time_samples)

    def set_time_step(self, time_step: int) -> None:
        """Set the current time step (for testing/debugging).
        
        Args:
            time_step: New time step value.
        """
        self._time_step = time_step
