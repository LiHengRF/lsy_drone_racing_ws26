"""
TrajectoryVisualizer
--------------------
Visualization for drone racing runs.

Key features
- Records position, velocity and instantaneous speed during flight
- Produces a figure (XY and XZ) with trajectory colored by speed
- Saves multiple episodes/runs without overwriting:
  - A time-stamped run folder is created at controller construction time
  - Each episode is saved as *_epXXX.png / *_epXXX_speed.png / *_epXXX.npz
- Updates gate/obstacle positions from nominal -> true when the object is within sensor range
  (objects are "locked" once confirmed true)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class TrajectoryVisualizerConfig:
    enabled: bool = True
    live: bool = False
    live_update_every: int = 10

    # Output
    out_dir: str = "trajectory_viz"
    file_prefix: str = "mpcc"

    # Geometry / rendering
    gate_bar_length_xy: float = 0.70          # [m] stylized gate bar length in XY
    obstacle_radius: float = 0.015            # [m] pole radius (diameter 0.03 m)
    obstacle_height: float = 1.52             # [m] pole height (top marker at this z)

    # Sensor-range logic
    sensor_margin: float = 0.05               # [m] to avoid flicker at boundary


def _safe_getattr(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get nested attributes from dot-access configs (OmegaConf, SimpleNamespace, dict)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _to_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if x is None:
        return np.zeros((0, 3), dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(dtype) if dtype is not None else x
    return np.array(x, dtype=dtype)


def _roman_numeral(n: int) -> str:
    """Return uppercase Roman numeral for 1 <= n <= 50 (enough for typical gate counts)."""
    vals = [
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out) if out else str(n)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def _square_corners(center: np.ndarray, u: np.ndarray, v: np.ndarray, half: float) -> np.ndarray:
    """Return 4x2 corners of a square centered at `center` with axes `u`, `v` and half-width `half`."""
    return np.array(
        [
            center + half * (u + v),
            center + half * (u - v),
            center - half * (u + v),
            center - half * (u - v),
        ]
    )


# Gate dimensions (square frame)
GATE_OUTER_WIDTH = 0.72  # [m]
GATE_INNER_WIDTH = 0.40  # [m]


class TrajectoryVisualizer:
    # Share a session directory and run counter across instances so multiple episodes
    # in the same process reuse the same timestamped folder.
    _SESSION_DIRS: dict[Path, Path] = {}
    _RUN_COUNTERS: dict[Path, int] = {}

    def __init__(
        self,
        config: Any,
        initial_obs: dict[str, np.ndarray],
        *,
        viz_cfg: Optional[TrajectoryVisualizerConfig] = None,
        title: str = "MPCC",
    ) -> None:
        self._title = title
        self._cfg = viz_cfg or self._cfg_from_level_config(config)

        self._env_freq: float = float(_safe_getattr(config, "env.freq", 50.0))
        self._sensor_range: Optional[float] = _safe_getattr(config, "env.sensor_range", None)
        if self._sensor_range is not None:
            self._sensor_range = float(self._sensor_range)

        # Output directories:
        # - A time-stamped *session* folder is created at controller construction time
        # - Inside that folder, each episode/run gets its own subfolder (run000, run001, ...)
        #   so file names do not need an _epXXX suffix.
        self._out_dir = Path(self._cfg.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        out_dir_key = self._out_dir.resolve()

        if out_dir_key not in self._SESSION_DIRS:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            session_dir = self._out_dir / ts
            session_dir.mkdir(parents=True, exist_ok=True)
            self._SESSION_DIRS[out_dir_key] = session_dir
            self._RUN_COUNTERS[out_dir_key] = 1  # start from run001

        self._session_dir = self._SESSION_DIRS[out_dir_key]
        self._out_dir_key = out_dir_key
        self._active_run_dir: Optional[Path] = None

        # Buffers
        self._pos_hist: list[np.ndarray] = []
        self._vel_hist: list[np.ndarray] = []
        self._speed_hist: list[float] = []

        # Object state (nominal, estimate, confirmed)
        self._gates_nom = np.zeros((0, 3), dtype=float)
        self._obst_nom = np.zeros((0, 3), dtype=float)
        self._gates_est = np.zeros((0, 3), dtype=float)
        self._obst_est = np.zeros((0, 3), dtype=float)
        self._gates_confirmed = np.zeros((0,), dtype=bool)
        self._obst_confirmed = np.zeros((0,), dtype=bool)

        # Gate orientation (optional)
        self._gates_quat_est: Optional[np.ndarray] = None

        # Live plotting handles (lazy)
        self._mpl_ready = False
        self._fig = None
        self._ax_xy = None
        self._ax_xz = None
        self._last_live_update_step = 0

        # Init from first obs
        self._needs_object_init = True
        self._init_objects_from_obs(config=config, obs=initial_obs)
        self._needs_object_init = False

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg_from_level_config(config: Any) -> TrajectoryVisualizerConfig:
        cfg = TrajectoryVisualizerConfig()

        def read(path1: str, path2: str, cast):
            val = _safe_getattr(config, path1, None)
            if val is None:
                val = _safe_getattr(config, path2, None)
            return cast(val) if val is not None else None

        enabled = read("controller.visualizer.enabled", "visualizer.enabled", bool)
        if enabled is not None:
            cfg.enabled = enabled

        live = read("controller.visualizer.live", "visualizer.live", bool)
        if live is not None:
            cfg.live = live

        upd = read("controller.visualizer.live_update_every", "visualizer.live_update_every", int)
        if upd is not None:
            cfg.live_update_every = upd

        out_dir = read("controller.visualizer.out_dir", "visualizer.out_dir", str)
        if out_dir is not None:
            cfg.out_dir = out_dir

        prefix = read("controller.visualizer.file_prefix", "visualizer.file_prefix", str)
        if prefix is not None:
            cfg.file_prefix = prefix

        obst_r = read("controller.visualizer.obstacle_radius", "visualizer.obstacle_radius", float)
        if obst_r is not None:
            cfg.obstacle_radius = float(obst_r)

        obst_h = read("controller.visualizer.obstacle_height", "visualizer.obstacle_height", float)
        if obst_h is not None:
            cfg.obstacle_height = float(obst_h)

        return cfg

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, *, new_episode: bool = True) -> None:
        """Clear telemetry buffers and start a new run subfolder.

        The controller may run multiple episodes within a single Python process.
        To avoid overwriting outputs while keeping clean file names, each episode/run
        is written to its own subfolder within the time-stamped session directory:
            <out_dir>/<timestamp>/run000/, run001/, ...

        Notes
        - Many frameworks call this from `episode_callback()` at the beginning of each episode.
        - If `reset()` is not called before `update()`, the visualizer will lazily create run000
          on the first `update()`/`finalize()`.
        """
        if not self._cfg.enabled:
            return

        if new_episode:
            # Defer opening the next run directory until we actually have data to save.
            self._active_run_dir = None
            self._needs_object_init = True

        self._pos_hist.clear()
        self._vel_hist.clear()
        self._speed_hist.clear()

    # ------------------------------------------------------------------
    # Runtime update
    # ------------------------------------------------------------------

    def _open_new_run_dir(self) -> None:
        """Create and switch to a new per-run subfolder under the session directory."""
        idx = self._RUN_COUNTERS[self._out_dir_key]
        run_name = f"run{idx:03d}"
        run_dir = self._session_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._active_run_dir = run_dir
        self._RUN_COUNTERS[self._out_dir_key] = idx + 1

    def _ensure_active_run_dir(self) -> None:
        """Ensure there is an active run directory (lazy init)."""
        if self._active_run_dir is None:
            self._open_new_run_dir()

    def update(self, *, obs: dict[str, np.ndarray], action: Any = None, info: Optional[dict] = None) -> None:
        if not self._cfg.enabled:
            return

        pos = _to_numpy(obs.get("pos", np.zeros(3)), float).reshape(-1)
        vel = _to_numpy(obs.get("vel", np.zeros(3)), float).reshape(-1)
        speed = float(np.linalg.norm(vel))

        # Initialize objects at episode start from first observation
        if getattr(self, "_needs_object_init", False):
            self._init_objects_from_obs(config=None, obs=obs)
            self._needs_object_init = False

        # Record telemetry
        self._pos_hist.append(pos.copy())
        self._vel_hist.append(vel.copy())
        self._speed_hist.append(speed)

        # Update object estimates from nominal -> true based on sensor_range
        self._update_objects_estimates(drone_pos=pos, obs=obs)

        # Optional live update
        if self._cfg.live:
            step = len(self._pos_hist)
            if step - self._last_live_update_step >= max(1, int(self._cfg.live_update_every)):
                self._last_live_update_step = step
                self._live_draw()

    # ------------------------------------------------------------------
    # Finalization / saving
    # ------------------------------------------------------------------

    def finalize(self, *, file_prefix: Optional[str] = None) -> dict[str, Path]:
        if not self._cfg.enabled:
            return {}

        prefix = file_prefix or self._cfg.file_prefix
        title_part = (self._title or "").strip().lower()
        base_name = prefix if file_prefix is not None or not title_part else f"{prefix}_{title_part}"
        self._ensure_active_run_dir()
        base = self._active_run_dir / base_name

        pos = np.vstack(self._pos_hist) if self._pos_hist else np.zeros((0, 3), dtype=float)
        vel = np.vstack(self._vel_hist) if self._vel_hist else np.zeros((0, 3), dtype=float)
        speed = np.array(self._speed_hist, dtype=float)

        data_path = base.with_suffix(".npz")
        np.savez_compressed(
            data_path,
            pos=pos,
            vel=vel,
            speed=speed,
            gates_nom=self._gates_nom,
            gates_est=self._gates_est,
            obstacles_nom=self._obst_nom,
            obstacles_est=self._obst_est,
        )

        fig_path = base.with_suffix(".png")
        speed_path = base.with_name(base.name + "_speed.png")

        self._plot_xy_xz(pos, speed, fig_path)
        self._plot_speed_time(speed, speed_path)

        return {"trajectory": fig_path, "speed": speed_path, "data": data_path}

    # ------------------------------------------------------------------
    # Object initialization & updates
    # ------------------------------------------------------------------

    def _init_objects_from_obs(self, *, config: Any, obs: dict[str, np.ndarray]) -> None:
        # Prefer track config nominal, otherwise fall back to observation
        gates_nom = self._read_gates_nominal_from_config(config) if config is not None else None
        obst_nom = self._read_obstacles_nominal_from_config(config) if config is not None else None

        self._gates_nom = gates_nom if gates_nom is not None else _to_numpy(obs.get("gates_pos", []), float)
        self._obst_nom = obst_nom if obst_nom is not None else _to_numpy(obs.get("obstacles_pos", []), float)

        self._gates_est = np.array(self._gates_nom, dtype=float, copy=True)
        self._obst_est = np.array(self._obst_nom, dtype=float, copy=True)

        self._gates_confirmed = np.zeros((len(self._gates_est),), dtype=bool)
        self._obst_confirmed = np.zeros((len(self._obst_est),), dtype=bool)

        gates_quat = obs.get("gates_quat", None)
        self._gates_quat_est = _to_numpy(gates_quat, float).copy() if gates_quat is not None else None

    def _update_objects_estimates(self, *, drone_pos: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        if self._sensor_range is None:
            # If sensor_range is unknown, accept observations as truth.
            if "gates_pos" in obs:
                self._gates_est = _to_numpy(obs["gates_pos"], float)
            if "obstacles_pos" in obs:
                self._obst_est = _to_numpy(obs["obstacles_pos"], float)
            return

        r = float(self._sensor_range) + float(self._cfg.sensor_margin)

        gates_obs = obs.get("gates_pos", None)
        if gates_obs is not None and self._gates_est.size != 0:
            g = _to_numpy(gates_obs, float)
            for i in range(min(len(g), len(self._gates_est))):
                if self._gates_confirmed[i]:
                    continue
                if float(np.linalg.norm(g[i] - drone_pos)) <= r:
                    self._gates_est[i] = g[i]
                    self._gates_confirmed[i] = True

        obst_obs = obs.get("obstacles_pos", None)
        if obst_obs is not None and self._obst_est.size != 0:
            o = _to_numpy(obst_obs, float)
            for i in range(min(len(o), len(self._obst_est))):
                if self._obst_confirmed[i]:
                    continue
                if float(np.linalg.norm(o[i] - drone_pos)) <= r:
                    self._obst_est[i] = o[i]
                    self._obst_confirmed[i] = True

        gates_quat = obs.get("gates_quat", None)
        if gates_quat is not None:
            self._gates_quat_est = _to_numpy(gates_quat, float).copy()

    @staticmethod
    def _read_gates_nominal_from_config(config: Any) -> Optional[np.ndarray]:
        gates = _safe_getattr(config, "env.track.gates", None)
        if gates is None:
            return None
        pts = []
        for g in gates:
            pos = _safe_getattr(g, "pos", None)
            if pos is not None:
                pts.append(pos)
        return _to_numpy(pts, float) if pts else None

    @staticmethod
    def _read_obstacles_nominal_from_config(config: Any) -> Optional[np.ndarray]:
        obst = _safe_getattr(config, "env.track.obstacles", None)
        if obst is None:
            return None
        pts = []
        for o in obst:
            pos = _safe_getattr(o, "pos", None)
            if pos is not None:
                pts.append(pos)
        return _to_numpy(pts, float) if pts else None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_xy_xz(self, pos: np.ndarray, speed: np.ndarray, out_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib import colors

        fig = plt.figure(figsize=(6.0, 7.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.30)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_xy)

        ax_xy.set_title(self._title)
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.grid(True, alpha=0.5)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.tick_params(axis="x", labelbottom=True)
        ax_xz.set_xlabel("x [m]")
        ax_xz.set_ylabel("z [m]")
        ax_xz.grid(True, alpha=0.5)
        ax_xz.set_aspect("equal", adjustable="box")

        lc_xy = None
        if pos.shape[0] >= 2:
            speed_min = 0.0
            speed_max = 2.3
            speed_norm = colors.PowerNorm(gamma=0.6, vmin=speed_min, vmax=speed_max)
            speed_cmap = plt.get_cmap("turbo")

            def colored_line(ax, x, y, c):
                pts = np.stack([x, y], axis=1)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(
                    segs, array=c[:-1], linewidths=3.0, cmap=speed_cmap, norm=speed_norm
                )
                ax.add_collection(lc)
                ax.autoscale()
                return lc

            lc_xy = colored_line(ax_xy, pos[:, 0], pos[:, 1], speed)
            colored_line(ax_xz, pos[:, 0], pos[:, 2], speed)

            # Start / end markers
            ax_xy.scatter(pos[0, 0], pos[0, 1], color="green", s=36, zorder=5, label="start")
            ax_xy.scatter(pos[-1, 0], pos[-1, 1], color="blue", s=40, zorder=5, label="end")
            ax_xz.scatter(pos[0, 0], pos[0, 2], color="green", s=36, zorder=5, label="start")
            ax_xz.scatter(pos[-1, 0], pos[-1, 2], color="blue", s=40, zorder=5, label="end")
        else:
            ax_xy.plot(pos[:, 0], pos[:, 1])
            ax_xz.plot(pos[:, 0], pos[:, 2])

        # Overlay objects
        self._draw_gates_xy(ax_xy)
        self._draw_obstacles_xy(ax_xy)
        self._draw_gates_xz(ax_xz)
        self._draw_obstacles_xz(ax_xz)

        ax_xy.set_ylim(-1.5, 1.5)
        ax_xz.set_ylim(0.0, 2.0)

        ax_xy.set_xlim(-1.75, 1.75)
        ax_xz.set_xlim(-1.75, 1.75)

        if lc_xy is not None:
            pos_xy = ax_xy.get_position()
            pos_xz = ax_xz.get_position()
            y0 = min(pos_xy.y0, pos_xz.y0)
            y1 = max(pos_xy.y1, pos_xz.y1)
            x1 = max(pos_xy.x1, pos_xz.x1)
            cbar_pad = 0.02
            cbar_width = 0.03
            cax = fig.add_axes([x1 + cbar_pad, y0, cbar_width, y1 - y0])
            cbar = fig.colorbar(lc_xy, cax=cax)
            cbar.set_label("speed [m/s]")

        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_speed_time(self, speed: np.ndarray, out_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(len(speed)) / max(1.0, float(self._env_freq))

        fig = plt.figure(figsize=(6.0, 2.6))
        ax = fig.add_subplot(111)
        ax.plot(t, speed, linewidth=2.0)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("speed [m/s]")
        ax.grid(True, alpha=0.5)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _draw_gates_xy(self, ax) -> None:
        if self._gates_est.size == 0:
            return
        for i, p in enumerate(self._gates_est):
            try:
                R = _quat_to_rotmat(self._gates_quat_est[i]) if self._gates_quat_est is not None else None
            except Exception:
                R = None

            # Use gate local Y axis (frame width direction) to draw a line, representing the vertical plane edge-on.
            v = R[[0, 1], 1] if R is not None else np.array([0.0, 1.0])
            if not np.isfinite(v).all():
                v = np.array([0.0, 1.0])

            n = np.linalg.norm(v)
            if n < 1e-6:
                v = np.array([0.0, 1.0])
                n = 1.0
            v = v / n
            half = 0.5 * GATE_OUTER_WIDTH
            delta = half * v
            ax.plot(
                [p[0] - delta[0], p[0] + delta[0]],
                [p[1] - delta[1], p[1] + delta[1]],
                linewidth=3.5,
                color="black",
                zorder=3,
            )
            ax.text(
                p[0],
                p[1],
                _roman_numeral(i + 1),
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    def _draw_gates_xz(self, ax) -> None:
        if self._gates_est.size == 0:
            return
        for i, p in enumerate(self._gates_est):
            try:
                R = _quat_to_rotmat(self._gates_quat_est[i]) if self._gates_quat_est is not None else None
            except Exception:
                R = None

            if R is not None and np.isfinite(R).all():
                y_axis = R[:, 1]
                z_axis = R[:, 2]
            else:
                y_axis = np.array([0.0, 1.0, 0.0])
                z_axis = np.array([0.0, 0.0, 1.0])

            if not np.isfinite(y_axis).all() or not np.isfinite(z_axis).all():
                y_axis = np.array([0.0, 1.0, 0.0])
                z_axis = np.array([0.0, 0.0, 1.0])

            y_norm = np.linalg.norm(y_axis)
            z_norm = np.linalg.norm(z_axis)
            if y_norm < 1e-6 or z_norm < 1e-6:
                y_axis = np.array([0.0, 1.0, 0.0])
                z_axis = np.array([0.0, 0.0, 1.0])
                y_norm = z_norm = 1.0

            y_axis = y_axis / y_norm
            z_axis = z_axis / z_norm

            u = y_axis[[0, 2]]
            v = z_axis[[0, 2]]

            outer = _square_corners(np.array([p[0], p[2]]), u, v, 0.5 * GATE_OUTER_WIDTH)
            inner = _square_corners(np.array([p[0], p[2]]), u, v, 0.5 * GATE_INNER_WIDTH)

            from matplotlib.patches import Polygon

            ax.add_patch(
                Polygon(outer, closed=True, fill=False, edgecolor="black", linewidth=2.3, zorder=3)
            )
            ax.add_patch(
                Polygon(inner, closed=True, fill=False, edgecolor="black", linewidth=1.8, zorder=3)
            )
            ax.text(
                p[0],
                p[2],
                _roman_numeral(i + 1),
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    def _draw_obstacles_xy(self, ax) -> None:
        """Draw obstacles as gray circles in XY."""
        if self._obst_est.size == 0:
            return
        from matplotlib.patches import Circle

        r = float(self._cfg.obstacle_radius)
        for p in self._obst_est:
            circ = Circle((p[0], p[1]), r, facecolor="0.75", edgecolor="0.40", linewidth=1.5, alpha=0.9, zorder=2)
            ax.add_patch(circ)
        for i, p in enumerate(self._obst_est):
            offset = r * 1.3
            ax.text(
                p[0] + offset,
                p[1] + offset,
                f"{i + 1}",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    def _draw_obstacles_xz(self, ax) -> None:
        """Draw obstacles in XZ as slender poles with a reflective marker on top."""
        if self._obst_est.size == 0:
            return
        from matplotlib.patches import Circle

        r = float(self._cfg.obstacle_radius)
        z0 = float(self._cfg.obstacle_height)
        marker_r = max(r * 1.8, r + 0.005)
        for p in self._obst_est:
            # Pole
            ax.plot([p[0], p[0]], [0.0, z0], linewidth=3.5, color="0.35", alpha=0.9, zorder=2)
            # Reflective marker on top
            marker = Circle((p[0], z0), marker_r, facecolor="0.85", edgecolor="0.25", linewidth=1.5, alpha=0.95, zorder=3)
            ax.add_patch(marker)
        for i, p in enumerate(self._obst_est):
            ax.text(
                p[0],
                z0,
                f"{i + 1}",
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=4,
            )

    # ------------------------------------------------------------------
    # Live plotting (optional)
    # ------------------------------------------------------------------

    def _live_draw(self) -> None:
        """Very lightweight live plot; intended for local GUI runs only."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib import colors
        except Exception:
            return

        if not self._mpl_ready:
            plt.ion()
            self._fig = plt.figure(figsize=(6.0, 7.0))
            gs = self._fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.30)
            self._ax_xy = self._fig.add_subplot(gs[0, 0])
            self._ax_xz = self._fig.add_subplot(gs[1, 0], sharex=self._ax_xy)
            self._mpl_ready = True

        pos = np.vstack(self._pos_hist) if self._pos_hist else np.zeros((0, 3), dtype=float)
        speed = np.array(self._speed_hist, dtype=float)

        self._ax_xy.cla()
        self._ax_xz.cla()

        self._ax_xy.set_title(self._title)
        self._ax_xy.set_xlabel("x [m]")
        self._ax_xy.set_ylabel("y [m]")
        self._ax_xy.grid(True, alpha=0.5)
        self._ax_xy.set_aspect("equal", adjustable="box")
        self._ax_xy.tick_params(axis="x", labelbottom=True)
        self._ax_xz.set_xlabel("x [m]")
        self._ax_xz.set_ylabel("z [m]")
        self._ax_xz.grid(True, alpha=0.5)
        self._ax_xz.set_aspect("equal", adjustable="box")

        if pos.shape[0] >= 2:
            speed_min = 0.0
            speed_max = 2.3
            speed_norm = colors.PowerNorm(gamma=0.6, vmin=speed_min, vmax=speed_max)
            speed_cmap = plt.get_cmap("turbo")
            pts_xy = np.stack([pos[:, 0], pos[:, 1]], axis=1)
            segs_xy = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
            lc_xy = LineCollection(
                segs_xy, array=speed[:-1], linewidths=3.0, cmap=speed_cmap, norm=speed_norm
            )
            self._ax_xy.add_collection(lc_xy)
            self._ax_xy.autoscale()
            pts_xz = np.stack([pos[:, 0], pos[:, 2]], axis=1)
            segs_xz = np.stack([pts_xz[:-1], pts_xz[1:]], axis=1)
            lc_xz = LineCollection(
                segs_xz, array=speed[:-1], linewidths=3.0, cmap=speed_cmap, norm=speed_norm
            )
            self._ax_xz.add_collection(lc_xz)
            self._ax_xz.autoscale()

            # Start / end markers
            self._ax_xy.scatter(pos[0, 0], pos[0, 1], color="green", s=36, zorder=5)
            self._ax_xy.scatter(pos[-1, 0], pos[-1, 1], color="blue", s=40, zorder=5)
            self._ax_xz.scatter(pos[0, 0], pos[0, 2], color="green", s=36, zorder=5)
            self._ax_xz.scatter(pos[-1, 0], pos[-1, 2], color="blue", s=40, zorder=5)
        else:
            self._ax_xy.plot(pos[:, 0], pos[:, 1])
            self._ax_xz.plot(pos[:, 0], pos[:, 2])

        self._draw_gates_xy(self._ax_xy)
        self._draw_obstacles_xy(self._ax_xy)
        self._draw_gates_xz(self._ax_xz)
        self._draw_obstacles_xz(self._ax_xz)

        self._ax_xy.set_ylim(-1.5, 1.5)
        self._ax_xz.set_ylim(0.0, 2.0)

        self._ax_xy.set_xlim(-1.75, 1.75)
        self._ax_xz.set_xlim(-1.75, 1.75)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
