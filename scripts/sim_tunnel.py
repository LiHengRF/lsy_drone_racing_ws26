"""MPCC++ Simulator with Tunnel Visualization

Usage:
    python scripts/sim_tunnel.py --config level2.toml --controller mpcc++.py -n 3 -r -t
"""
from __future__ import annotations
import logging
import math
from pathlib import Path
import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller, draw_line
from lsy_drone_racing.utils.tunnel_visualizer import draw_tunnel

logger = logging.getLogger(__name__)


def _thin_polyline(points, max_segments):
    if points is None or len(points) <= 2:
        return points
    segs = len(points) - 1
    if segs <= max_segments:
        return points
    stride = math.ceil(segs / max_segments)
    thinned = points[::stride]
    if not np.allclose(thinned[-1], points[-1]):
        thinned = np.vstack([thinned, points[-1]])
    return thinned


def _sample_trajectory(ctrl):
    if hasattr(ctrl, "arc_trajectory"):
        try:
            arc = ctrl.arc_trajectory
            s = np.linspace(0.0, float(arc.x[-1]), 200)
            pts = np.asarray(arc(s), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3:
                return pts
        except Exception:
            pass
    return None


def simulate(
    config: str = "level0.toml",
    controller: str = None,
    n: int = 1,
    r: bool = False,
    t: bool = False,
):
    """
    Run simulation.
    
    Args:
        config: Config file
        controller: Controller file
        n: Number of runs
        r: Render GUI
        t: Show tunnel
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = r

    ctrl_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    ctrl_file = ctrl_path / (controller or cfg.controller.file)
    ctrl_cls = load_controller(ctrl_file)

    env = gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range, control_mode=cfg.env.control_mode,
        track=cfg.env.track, disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"), seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    times = []
    for run in range(n):
        obs, info = env.reset()
        ctrl = ctrl_cls(obs, info, cfg)
        i, fps = 0, 60

        while True:
            curr_time = i / cfg.env.freq
            action = np.asarray(jp.asarray(ctrl.compute_control(obs, info)), copy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)

            if terminated or truncated or done:
                break

            if r and ((i * fps) % cfg.env.freq) < fps:
                try:
                    # Draw trajectory
                    traj = _sample_trajectory(ctrl)
                    if traj is not None:
                        draw_line(env, _thin_polyline(traj, 400),
                                 rgba=np.array([0.3, 0.9, 0.3, 0.7]), min_size=1.5, max_size=1.5)
                    # Draw tunnel
                    if t:
                        draw_tunnel(env, ctrl)
                except RuntimeError:
                    pass
                env.render()
            i += 1

        ctrl.episode_callback()
        gates = obs["target_gate"]
        if gates == -1:
            gates = len(cfg.env.track.gates)
        finished = gates == len(cfg.env.track.gates)
        logger.info(f"Run {run+1}: {curr_time:.2f}s, finished={finished}, gates={gates}")
        ctrl.episode_reset()
        times.append(curr_time if finished else None)

    env.close()
    completed = sum(x is not None for x in times)
    print(f"\nCompleted: {completed}/{n}")
    return times


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)