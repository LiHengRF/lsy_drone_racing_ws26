"""
MPCC++ Controller for Drone Racing with Tunnel Constraints - FIXED VERSION v2

修复内容:
1. 滑动窗口tunnel参数 - 随无人机进度滑动更新
2. 改进的初始化 - 第一次求解时更好的warm start
3. 每次控制循环更新参数
4. [NEW] v_theta_cmd 最小值约束 - 强制前进
5. [NEW] episode_callback 中重新初始化
6. [NEW] 检测门/障碍物位置跳变 - 不仅仅是visited flags
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else, log, exp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# Import path planning module
from lsy_drone_racing.control.path_planning import PathPlanner, PathConfig, PathVisualizer, VISUALIZER_AVAILABLE

# Import drone racing framework
from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MPCCConfig:
    """Configuration for MPCC++ controller with tunnel constraints."""
    # MPC Horizon
    N_horizon: int = 40
    T_horizon: float = 0.7
    
    # Arc-length model
    model_arc_step: float = 0.05
    model_traj_length: float = 12.0
    
    # Cost function weights
    q_lag: float = 80.0                    # Lag error weight
    q_lag_peak: float = 500.0              # Lag error weight at gates
    q_contour: float = 120.0               # Contour error weight
    q_contour_peak: float = 700.0          # Contour error weight at gates
    q_attitude: float = 1.0                # Attitude regularization
    
    # Control smoothness
    r_thrust: float = 0.1                  # Thrust rate penalty
    r_roll: float = 0.3                    # Roll rate penalty
    r_pitch: float = 0.3                   # Pitch rate penalty
    r_yaw: float = 0.50                     # Yaw rate penalty
    
    # Speed incentive
    mu_speed: float = 4.0
    w_speed_gate: float = 6.0
    
    # Attitude constraints
    roll_limit: float = 0.7
    pitch_limit: float = 0.7
    
    # ==========================================================================
    # Tunnel (Spatial Constraint) Parameters
    # ==========================================================================
    tunnel_radius_far: float = 0.45
    tunnel_radius_gate: float = 0.20
    tunnel_radius_obstacle: float = 0.20
    
    tunnel_w_tunnel: float = 30.0
    alpha_tunnel: float = 40.0
    
    tunnel_gate_sigma: float = 8.0
    tunnel_obstacle_sigma: float = 8.0
    
    # Safety bounds
    pos_bounds: tuple = (
        (-3.0, 3.0),
        (-2.5, 2.5),
        (-0.2, 2.5),
    )
    vel_bounds: tuple = (-1.0, 8.0)
    
    # Path planning
    planned_duration: float = 30.0
    
    # Visualization settings
    visualization_enabled: bool = False
    visualization_width: int = 1400
    visualization_height: int = 1000
    visualization_output_dir: Optional[str] = None
    log_interval: int = 20
    
    # ==========================================================================
    # Parameter update settings
    # ==========================================================================
    update_params_every_step: bool = True
    theta_lookahead_margin: float = 0.5
    
    # ==========================================================================
    # [NEW] v_theta_cmd 最小值 - 强制前进，避免"原地不动"的局部最优
    # ==========================================================================
    v_theta_min: float = 0.05  # 最小前进速度 (m/s along path)
    v_theta_max: float = 1.0   # 最大前进速度
    
    # ==========================================================================
    # [NEW] 门/障碍物位置跳变检测阈值
    # ==========================================================================
    gate_pos_change_thresh: float = 0.05   # [m] 门位置变化阈值
    obst_pos_change_thresh: float = 0.05   # [m] 障碍物位置变化阈值


# =============================================================================
# MPCC++ Controller with Tunnel Constraints - FIXED v2
# =============================================================================

class MPCCController(Controller):
    """
    Model Predictive Contouring Control with Tunnel Constraints - FIXED VERSION v2.
    
    Key fixes:
    1. Sliding window for tunnel parameters
    2. Improved initialization with proper warm start
    3. Minimum v_theta_cmd constraint to prevent "stuck" solutions
    4. Episode reset properly reinitializes trajectory and warm start
    5. Detects gate/obstacle POSITION changes, not just visited flags
    """
    
    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        mpcc_config: Optional[MPCCConfig] = None,
        path_config: Optional[PathConfig] = None
    ):
        """Initialize the MPCC++ controller."""
        super().__init__(obs, info, config)
        
        # Configurations
        self.mpcc_cfg = mpcc_config or MPCCConfig()
        self.path_cfg = path_config or PathConfig()
        
        # Controller state
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self.finished = False
        
        # [NEW] Flag for episode reset re-initialization
        self._need_reinit_on_next_step = False
        
        # Load dynamics parameters
        self._dyn_params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(self._dyn_params["mass"])
        self._gravity = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = self._mass * self._gravity
        
        # Initialize path planner
        self.path_planner = PathPlanner(self.path_cfg)
        
        # Store initial position
        self._initial_pos = obs["pos"].copy()
        
        # Environment change detection - flags
        self._last_gate_flags = None
        self._last_obst_flags = None
        
        # [NEW] Environment change detection - positions
        self._last_gate_positions = obs["gates_pos"].copy()
        self._last_obstacle_positions = obs["obstacles_pos"].copy()
        
        # Gate detection tracking
        num_gates = len(obs['gates_pos'])
        self._gate_detected_flags = np.zeros(num_gates, dtype=bool)
        self._gate_real_positions = np.full((num_gates, 3), np.nan)
        
        # Plan initial trajectory
        self._plan_trajectory(obs)
        
        # MPC parameters
        self.N = self.mpcc_cfg.N_horizon
        self.T = self.mpcc_cfg.T_horizon
        self.dt = self.T / self.N
        self.model_arc_step = self.mpcc_cfg.model_arc_step
        self.model_traj_length = self.mpcc_cfg.model_traj_length
        
        # Track absolute theta and window offset
        self.absolute_theta = 0.0
        self.theta_window_start = 0.0
        
        # Build MPCC solver
        self._build_solver()
        
        # Initialize control states
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        
        # Current observation
        self._current_pos = obs["pos"].copy()
        
        # Initialize warm start properly
        self._initialize_warm_start(obs)
        
        # Initialize visualization
        self.visualizer = None
        if self.mpcc_cfg.visualization_enabled:
            self.visualizer = PathVisualizer(
                width=self.mpcc_cfg.visualization_width,
                height=self.mpcc_cfg.visualization_height,
                title="MPCC++ Drone Racing - Trajectory Visualization",
                output_dir=self.mpcc_cfg.visualization_output_dir,
                enabled=True
            )
            if self.visualizer.is_available:
                self.visualizer.visualize_trajectory(
                    self._trajectory_result,
                    drone_position=obs['pos'],
                    gate_detected_status=self._gate_detected_flags,
                    show=True
                )
        
        print(f"[MPCC++] Initialized with sliding window tunnel constraints (v2).")
        print(f"[MPCC++] Horizon: N={self.N}, T={self.T:.2f}s")
        print(f"[MPCC++] v_theta bounds: [{self.mpcc_cfg.v_theta_min}, {self.mpcc_cfg.v_theta_max}]")
        print(f"[MPCC++] Position change thresholds: gate={self.mpcc_cfg.gate_pos_change_thresh}m, "
              f"obst={self.mpcc_cfg.obst_pos_change_thresh}m")
        print(f"[MPCC++] Arc trajectory length: {self.arc_trajectory.x[-1]:.2f}")
    
    # =========================================================================
    # Improved Warm Start Initialization
    # =========================================================================
    
    def _initialize_warm_start(self, obs: dict[str, NDArray[np.floating]]):
        """Initialize warm start states for first solver call."""
        quat = obs["quat"]
        roll, pitch, yaw = Rotation.from_quat(quat).as_euler("xyz")
        
        closest_theta, _ = self.path_planner.find_closest_point(
            self.arc_trajectory, 
            obs["pos"],
            sample_interval=self.model_arc_step
        )
        
        self.absolute_theta = max(0.0, closest_theta)
        self.theta_window_start = max(0.0, self.absolute_theta - self.mpcc_cfg.theta_lookahead_margin)
        self.last_theta = self.absolute_theta - self.theta_window_start
        
        self._x_warm = []
        self._u_warm = []
        
        x_init = np.concatenate([
            obs["pos"],
            obs["vel"],
            np.array([roll, pitch, yaw]),
            np.array([self.hover_thrust, self.hover_thrust]),
            np.zeros(3),
            np.array([self.last_theta])
        ])
        
        max_arc = float(self.arc_trajectory.x[-1])
        for i in range(self.N + 1):
            t_ahead = i * self.dt
            theta_ahead = self.last_theta + t_ahead * self.mpcc_cfg.v_theta_min * 2
            theta_ahead = min(theta_ahead, self.model_traj_length - 0.1)
            
            abs_theta = self.theta_window_start + theta_ahead
            abs_theta = min(abs_theta, max_arc)
            target_pos = self.arc_trajectory(abs_theta)
            
            x_warm_i = x_init.copy()
            alpha = min(1.0, t_ahead / self.T)
            x_warm_i[:3] = (1 - alpha) * obs["pos"] + alpha * target_pos
            x_warm_i[14] = theta_ahead
            
            self._x_warm.append(x_warm_i)
        
        for i in range(self.N):
            u_warm_i = np.array([0.0, 0.0, 0.0, 0.0, self.mpcc_cfg.v_theta_min * 2])
            self._u_warm.append(u_warm_i)
        
        param_vec = self._encode_trajectory_params(self.theta_window_start)
        for k in range(self.N + 1):
            self.solver.set(k, "p", param_vec)
        
        print(f"[MPCC++] Initialized warm start: absolute_theta={self.absolute_theta:.3f}, "
              f"window_start={self.theta_window_start:.3f}, relative_theta={self.last_theta:.3f}")
    
    # =========================================================================
    # Trajectory Planning
    # =========================================================================
    
    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Plan or replan the trajectory."""
        print(f"[MPCC++] Planning trajectory at T={self._step_count / self._ctrl_freq:.2f}s")
        
        result = self.path_planner.plan_trajectory(
            obs,
            trajectory_duration=self.mpcc_cfg.planned_duration,
            sampling_freq=self._ctrl_freq,
            for_mpcc=True,
            mpcc_extension_length=self.mpcc_cfg.model_traj_length
        )
        
        self._trajectory_result = result
        self.trajectory = result.spline
        self.arc_trajectory = result.arc_spline
        self.waypoints = result.waypoints
        self.total_arc_length = result.total_length
        
        self._cached_gate_centers = obs["gates_pos"].copy()
        self._cached_obstacles = obs["obstacles_pos"].copy()
    
    # =========================================================================
    # MPCC Solver Construction - [MODIFIED] v_theta_min constraint
    # =========================================================================
    
    def _build_solver(self):
        """Build the acados MPCC solver with tunnel constraints."""
        model = self._build_dynamics_model()
        
        ocp = AcadosOcp()
        ocp.model = model
        
        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = self.N
        
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = self._build_cost_expression()
        
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0
        cfg = self.mpcc_cfg
        
        ocp.constraints.lbx = np.array([
            thrust_min, thrust_min,
            -cfg.roll_limit, -cfg.pitch_limit, -1.57,
            -cfg.roll_limit, -cfg.pitch_limit
        ])
        ocp.constraints.ubx = np.array([
            thrust_max, thrust_max,
            cfg.roll_limit, cfg.pitch_limit, 1.57,
            cfg.roll_limit, cfg.pitch_limit
        ])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 6, 7])
        
        # [MODIFIED] Input constraints - 强制最小前进速度
        ocp.constraints.lbu = np.array([
            -10.0, -10.0, -10.0, -10.0, 
            cfg.v_theta_min  # 强制最小前进速度
        ])
        ocp.constraints.ubu = np.array([
            10.0, 10.0, 10.0, 10.0, 
            cfg.v_theta_max
        ])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        
        ocp.constraints.x0 = np.zeros(self.nx)
        
        param_vec = self._encode_trajectory_params(0.0)
        ocp.parameter_values = param_vec
        
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = self.T
        
        self.solver = AcadosOcpSolver(ocp, json_file="mpcc_racing.json", verbose=False)
        self.ocp = ocp
    
    def _build_dynamics_model(self) -> AcadosModel:
        """Build the quadrotor dynamics model."""
        model_name = "mpcc_drone_racing_tunnel"
        
        mass = self._mass
        gravity = self._gravity
        
        params_pitch_rate = [-6.003842038081178, 6.213752925707588]
        params_roll_rate = [-3.960889336015948, 4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]
        
        self.px = MX.sym("px")
        self.py = MX.sym("py")
        self.pz = MX.sym("pz")
        self.vx = MX.sym("vx")
        self.vy = MX.sym("vy")
        self.vz = MX.sym("vz")
        self.roll = MX.sym("roll")
        self.pitch = MX.sym("pitch")
        self.yaw = MX.sym("yaw")
        self.f_collective = MX.sym("f_collective")
        self.f_cmd = MX.sym("f_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")
        self.theta = MX.sym("theta")
        
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")
        
        states = vertcat(
            self.px, self.py, self.pz,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.f_collective, self.f_cmd,
            self.r_cmd, self.p_cmd, self.y_cmd,
            self.theta
        )
        inputs = vertcat(
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd,
            self.v_theta_cmd
        )
        
        thrust = self.f_collective
        inv_mass = 1.0 / mass
        
        ax = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * cos(self.yaw)
            + sin(self.roll) * sin(self.yaw)
        )
        ay = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * sin(self.yaw)
            - sin(self.roll) * cos(self.yaw)
        )
        az = inv_mass * thrust * cos(self.roll) * cos(self.pitch) - gravity
        
        f_dyn = vertcat(
            self.vx, self.vy, self.vz,
            ax, ay, az,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_cmd - self.f_collective),
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd,
            self.v_theta_cmd
        )
        
        n_samples = int(self.model_traj_length / self.model_arc_step)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)
        self.qc_dyn = MX.sym("qc_dyn", n_samples)
        self.r_tunnel = MX.sym("r_tunnel", n_samples)
        
        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn, self.r_tunnel)
        
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params
        
        return model
    
    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        """CasADi-friendly linear interpolation."""
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        
        idx_low = floor(idx_float)
        idx_high = idx_low + 1
        alpha = idx_float - idx_low
        
        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_high >= M, M - 1, idx_high)
        
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        
        return (1.0 - alpha) * p_low + alpha * p_high
    
    def _encode_trajectory_params(self, theta_offset: float = 0.0) -> np.ndarray:
        """Encode trajectory for MPCC cost function with sliding window."""
        cfg = self.mpcc_cfg
        
        theta_samples_relative = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        theta_samples_absolute = theta_samples_relative + theta_offset
        
        max_theta = float(self.arc_trajectory.x[-1])
        theta_samples_absolute = np.clip(theta_samples_absolute, 0.0, max_theta)
        
        pd_vals = self.arc_trajectory(theta_samples_absolute)
        tp_vals = self.arc_trajectory.derivative(1)(theta_samples_absolute)
        
        qc_dyn = np.zeros_like(theta_samples_relative)
        
        for gate_center in self._cached_gate_centers:
            d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
            qc_gate = 0.4 * np.exp(-cfg.tunnel_gate_sigma * d_gate**2)
            qc_dyn = np.maximum(qc_dyn, qc_gate)
        
        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            qc_obs = 0.2 * np.exp(-cfg.tunnel_obstacle_sigma * d_obs_xy**2)
            qc_dyn = np.maximum(qc_dyn, qc_obs)
        
        r_tunnel = np.full_like(theta_samples_relative, cfg.tunnel_radius_far, dtype=float)
        
        for gate_center in self._cached_gate_centers:
            d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
            gate_influence = np.exp(-cfg.tunnel_gate_sigma * d_gate**2)
            r_gate_profile = (
                cfg.tunnel_radius_far
                - (cfg.tunnel_radius_far - cfg.tunnel_radius_gate) * gate_influence
            )
            r_tunnel = np.minimum(r_tunnel, r_gate_profile)
        
        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            obs_influence = np.exp(-cfg.tunnel_obstacle_sigma * d_obs_xy**2)
            r_obs_profile = (
                cfg.tunnel_radius_far
                - (cfg.tunnel_radius_far - cfg.tunnel_radius_obstacle) * obs_influence
            )
            r_tunnel = np.minimum(r_tunnel, r_obs_profile)
        
        return np.concatenate([
            pd_vals.reshape(-1),
            tp_vals.reshape(-1),
            qc_dyn,
            r_tunnel
        ])
    
    def _build_cost_expression(self):
        """Build MPCC stage cost expression with tunnel soft-constraint."""
        cfg = self.mpcc_cfg
        
        position = vertcat(self.px, self.py, self.pz)
        attitude = vertcat(self.roll, self.pitch, self.yaw)
        control = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_dyn, dim=1)
        r_tunnel_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.r_tunnel, dim=1)
        
        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit
        e_contour = e_theta - e_lag
        
        Q_w = cfg.q_attitude * DM(np.eye(3))
        track_cost = (
            (cfg.q_lag + cfg.q_lag_peak * qc_theta) * dot(e_lag, e_lag)
            + (cfg.q_contour + cfg.q_contour_peak * qc_theta) * dot(e_contour, e_contour)
            + attitude.T @ Q_w @ attitude
        )
        
        R_df = DM(np.diag([cfg.r_thrust, cfg.r_roll, cfg.r_pitch, cfg.r_yaw]))
        smooth_cost = control.T @ R_df @ control
        
        speed_cost = -cfg.mu_speed * self.v_theta_cmd + cfg.w_speed_gate * qc_theta * (self.v_theta_cmd**2)
        
        e_contour_norm = norm_2(e_contour)
        margin = r_tunnel_theta - e_contour_norm
        tunnel_penalty = cfg.tunnel_w_tunnel * log(1.0 + exp(-cfg.alpha_tunnel * margin))
        
        return track_cost + smooth_cost + speed_cost + tunnel_penalty
    
    # =========================================================================
    # [MODIFIED] Environment Change Detection - 检测位置跳变
    # =========================================================================
    
    def _detect_environment_change(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """
        Detect changes in gate/obstacle - both visited flags AND position jumps.
        
        Key improvement: Detects when gate positions change (e.g., from nominal 
        to real position when drone approaches within 0.7m), not just when 
        gates_visited flag changes.
        """
        cfg = self.mpcc_cfg
        
        # 1. Flags logic
        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)
        
        if self._last_gate_flags is None:
            self._last_gate_flags = curr_gates.copy()
            self._last_obst_flags = curr_obst.copy()
        
        gate_flag_trigger = False
        obst_flag_trigger = False
        
        if curr_gates.shape == self._last_gate_flags.shape:
            gate_flag_trigger = np.any((~self._last_gate_flags) & curr_gates)
        if curr_obst.shape == self._last_obst_flags.shape:
            obst_flag_trigger = np.any((~self._last_obst_flags) & curr_obst)
        
        for i, is_visited in enumerate(curr_gates):
            if is_visited and not self._gate_detected_flags[i]:
                self._gate_detected_flags[i] = True
                self._gate_real_positions[i] = obs['gates_pos'][i]
                print(f"[GATE VISITED] Gate {i+1} passed at position: "
                      f"[{obs['gates_pos'][i][0]:.3f}, {obs['gates_pos'][i][1]:.3f}, {obs['gates_pos'][i][2]:.3f}]")
                
                if self.visualizer and self.visualizer.is_available:
                    self.visualizer.update_gate_detection(i, True, obs['gates_pos'][i])
        
        # 2. [NEW] Position jump detection
        gate_pos_trigger = False
        obst_pos_trigger = False
        
        if "gates_pos" in obs:
            curr_gate_pos = np.array(obs["gates_pos"], dtype=float)
            if curr_gate_pos.shape == self._last_gate_positions.shape:
                shifts = np.linalg.norm(curr_gate_pos - self._last_gate_positions, axis=1)
                changed_gates = np.where(shifts > cfg.gate_pos_change_thresh)[0]
                if len(changed_gates) > 0:
                    gate_pos_trigger = True
                    for idx in changed_gates:
                        print(f"[GATE POS CHANGE] Gate {idx+1} position shifted by {shifts[idx]:.3f}m: "
                              f"{self._last_gate_positions[idx]} -> {curr_gate_pos[idx]}")
            self._last_gate_positions = curr_gate_pos.copy()
        
        if "obstacles_pos" in obs:
            curr_obst_pos = np.array(obs["obstacles_pos"], dtype=float)
            if curr_obst_pos.shape == self._last_obstacle_positions.shape:
                shifts = np.linalg.norm(curr_obst_pos - self._last_obstacle_positions, axis=1)
                changed_obsts = np.where(shifts > cfg.obst_pos_change_thresh)[0]
                if len(changed_obsts) > 0:
                    obst_pos_trigger = True
                    for idx in changed_obsts:
                        print(f"[OBST POS CHANGE] Obstacle {idx+1} position shifted by {shifts[idx]:.3f}m")
            self._last_obstacle_positions = curr_obst_pos.copy()
        
        self._last_gate_flags = curr_gates.copy()
        self._last_obst_flags = curr_obst.copy()
        
        return bool(gate_flag_trigger or obst_flag_trigger or gate_pos_trigger or obst_pos_trigger)
    
    # =========================================================================
    # Safety Checks
    # =========================================================================
    
    def _check_position_bounds(self, pos: NDArray[np.floating]) -> bool:
        bounds = self.mpcc_cfg.pos_bounds
        for i, (low, high) in enumerate(bounds):
            if pos[i] < low or pos[i] > high:
                return False
        return True
    
    def _check_velocity_bounds(self, vel: NDArray[np.floating]) -> bool:
        speed = np.linalg.norm(vel)
        low, high = self.mpcc_cfg.vel_bounds
        return low < speed < high
    
    # =========================================================================
    # Main Control Loop
    # =========================================================================
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control command using MPCC++ with sliding window."""
        
        # [NEW] Handle episode reset re-initialization
        if self._need_reinit_on_next_step:
            print("[MPCC++] Re-initializing after episode reset...")
            self._plan_trajectory(obs)
            self._initialize_warm_start(obs)
            
            self._last_gate_positions = obs["gates_pos"].copy()
            self._last_obstacle_positions = obs["obstacles_pos"].copy()
            
            self._need_reinit_on_next_step = False
        
        self._current_pos = obs["pos"].copy()
        
        # Check for environment changes -> full replan
        if self._detect_environment_change(obs):
            print(f"[MPCC++] Environment change detected, replanning...")
            self._plan_trajectory(obs)
            
            param_vec = self._encode_trajectory_params(self.theta_window_start)
            for k in range(self.N + 1):
                self.solver.set(k, "p", param_vec)
        
        # Update sliding window parameters EVERY step
        if self.mpcc_cfg.update_params_every_step:
            new_window_start = max(0.0, self.absolute_theta - self.mpcc_cfg.theta_lookahead_margin)
            
            if abs(new_window_start - self.theta_window_start) > self.model_arc_step:
                self.theta_window_start = new_window_start
                
                param_vec = self._encode_trajectory_params(self.theta_window_start)
                for k in range(self.N + 1):
                    self.solver.set(k, "p", param_vec)
                
                if hasattr(self, "_x_warm") and self._x_warm is not None:
                    for i in range(len(self._x_warm)):
                        old_rel_theta = self._x_warm[i][14]
                        self._x_warm[i][14] = max(0.0, min(old_rel_theta, self.model_traj_length - 0.1))
        
        quat = obs["quat"]
        roll, pitch, yaw = Rotation.from_quat(quat).as_euler("xyz")
        
        relative_theta = self.absolute_theta - self.theta_window_start
        relative_theta = max(0.0, min(relative_theta, self.model_traj_length - 0.1))
        
        x_now = np.concatenate([
            obs["pos"],
            obs["vel"],
            np.array([roll, pitch, yaw]),
            np.array([self.last_f_collective, self.last_f_cmd]),
            self.last_rpy_cmd,
            np.array([relative_theta])
        ])
        
        if not hasattr(self, "_x_warm") or self._x_warm is None:
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]
        
        for i in range(self.N):
            self.solver.set(i, "x", self._x_warm[i])
            self.solver.set(i, "u", self._u_warm[i])
        self.solver.set(self.N, "x", self._x_warm[self.N])
        
        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)
        
        max_theta = float(self.arc_trajectory.x[-1])
        if self.absolute_theta >= max_theta - 0.5:
            self.finished = True
            print("[MPCC++] Finished: reached end of path")
        
        if not self._check_position_bounds(obs["pos"]):
            self.finished = True
            print("[MPCC++] Finished: position out of bounds")
        
        if not self._check_velocity_bounds(obs["vel"]):
            self.finished = True
            print("[MPCC++] Finished: velocity out of bounds")
        
        status = self.solver.solve()
        if status != 0:
            print(f"[MPCC++] WARNING: Solver returned status {status}")
        
        self._x_warm = [self.solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.solver.get(i, "u") for i in range(self.N)]
        
        x_next = self.solver.get(1, "x")
        
        self.last_f_collective = float(x_next[9])
        self.last_f_cmd = float(x_next[10])
        self.last_rpy_cmd = np.array(x_next[11:14])
        
        new_relative_theta = float(x_next[14])
        theta_progress = new_relative_theta - relative_theta
        self.absolute_theta += theta_progress
        self.last_theta = new_relative_theta
        
        cmd = np.array([
            self.last_rpy_cmd[0],
            self.last_rpy_cmd[1],
            self.last_rpy_cmd[2],
            self.last_f_cmd
        ], dtype=np.float32)
        
        if self.visualizer and self.visualizer.is_available:
            self.visualizer.update(
                drone_position=obs['pos'],
                gate_detected_status=self._gate_detected_flags,
                gate_real_positions=self._gate_real_positions
            )
        
        if self._step_count % self.mpcc_cfg.log_interval == 0:
            print(f"[MPCC++] T={self._step_count / self._ctrl_freq:.2f}s | "
                  f"z={obs['pos'][2]:.2f}m | "
                  f"thrust={self.last_f_cmd:.1f}N | "
                  f"abs_theta={self.absolute_theta:.2f}/{max_theta:.2f} | "
                  f"window=[{self.theta_window_start:.2f}, {self.theta_window_start + self.model_traj_length:.2f}] | "
                  f"status={status}")
        
        self._step_count += 1
        return cmd
    
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        return self.finished
    
    # =========================================================================
    # [MODIFIED] Episode Callback - 设置重新初始化标志
    # =========================================================================
    
    def episode_callback(self):
        """Called at episode reset."""
        print("[MPCC++] Episode reset - will reinitialize on next compute_control")
        self._step_count = 0
        self.finished = False
        
        self.absolute_theta = 0.0
        self.theta_window_start = 0.0
        
        self._x_warm = None
        self._u_warm = None
        self._last_gate_flags = None
        self._last_obst_flags = None
        
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        
        if hasattr(self, '_gate_detected_flags'):
            self._gate_detected_flags[:] = False
            self._gate_real_positions[:] = np.nan
        
        # [NEW] Set flag to reinitialize on next compute_control call
        self._need_reinit_on_next_step = True
    
    # =========================================================================
    # Debug and Visualization
    # =========================================================================
    
    def get_debug_lines(self):
        debug_lines = []
        
        if hasattr(self, "arc_trajectory"):
            try:
                full_path = self.arc_trajectory(self.arc_trajectory.x)
                debug_lines.append(
                    (full_path, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0)
                )
            except Exception:
                pass
        
        if hasattr(self, "_x_warm") and self._x_warm is not None:
            try:
                pred_states = np.array([x[:3] for x in self._x_warm])
                debug_lines.append(
                    (pred_states, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0)
                )
            except Exception:
                pass
        
        if hasattr(self, "absolute_theta") and hasattr(self, "arc_trajectory"):
            try:
                target = self.arc_trajectory(min(self.absolute_theta, self.arc_trajectory.x[-1]))
                segment = np.stack([self._current_pos, target])
                debug_lines.append(
                    (segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0)
                )
            except Exception:
                pass
        
        return debug_lines
    
    def get_trajectory(self) -> CubicSpline:
        return self.trajectory
    
    def get_arc_trajectory(self) -> CubicSpline:
        return self.arc_trajectory
    
    def get_progress(self) -> float:
        if hasattr(self, "arc_trajectory"):
            return self.absolute_theta / self.arc_trajectory.x[-1]
        return 0.0
    
    def get_tunnel_radius_profile(self) -> tuple:
        if not hasattr(self, 'arc_trajectory'):
            return None, None
        
        cfg = self.mpcc_cfg
        theta_samples = np.arange(
            self.theta_window_start, 
            self.theta_window_start + self.model_traj_length, 
            self.model_arc_step
        )
        theta_samples = np.clip(theta_samples, 0, self.arc_trajectory.x[-1])
        pd_vals = self.arc_trajectory(theta_samples)
        
        r_tunnel = np.full_like(theta_samples, cfg.tunnel_radius_far, dtype=float)
        
        for gate_center in self._cached_gate_centers:
            d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
            gate_influence = np.exp(-cfg.tunnel_gate_sigma * d_gate**2)
            r_gate_profile = (
                cfg.tunnel_radius_far
                - (cfg.tunnel_radius_far - cfg.tunnel_radius_gate) * gate_influence
            )
            r_tunnel = np.minimum(r_tunnel, r_gate_profile)
        
        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            obs_influence = np.exp(-cfg.tunnel_obstacle_sigma * d_obs_xy**2)
            r_obs_profile = (
                cfg.tunnel_radius_far
                - (cfg.tunnel_radius_far - cfg.tunnel_radius_obstacle) * obs_influence
            )
            r_tunnel = np.minimum(r_tunnel, r_obs_profile)
        
        return theta_samples, r_tunnel