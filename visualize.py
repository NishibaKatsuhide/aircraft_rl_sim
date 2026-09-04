from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ray
import ray.rllib  # noqa: F401
from ray.rllib.algorithms.algorithm import Algorithm

from aircraft_env import AircraftObstacleEnv


def get_action(algo, obs):
    # Compatibility path supported by current RLlib.
    return algo.compute_single_action(obs, explore=False)


def run_episode(algo, seed: int):
    env = AircraftObstacleEnv()
    obs, info = env.reset(seed=seed)

    done = False
    while not done:
        action = get_action(algo, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env


def _draw_animation(traj, obstacles, goal, threshold, save_path: str | None = None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for x, y, r in obstacles:
        z = np.linspace(0, 600, 16)
        theta = np.linspace(0, 2 * np.pi, 24)
        zz, tt = np.meshgrid(z, theta)
        xx = x + r * np.cos(tt)
        yy = y + r * np.sin(tt)
        ax.plot_surface(xx, yy, zz, alpha=0.10, linewidth=0)

    theta = np.linspace(0, 2 * np.pi, 48)
    z_cyl = np.linspace(0, 600, 16)
    zz, tt = np.meshgrid(z_cyl, theta)
    xx = goal[0] + threshold * np.cos(tt)
    yy = goal[1] + threshold * np.sin(tt)
    ax.plot_surface(xx, yy, zz, alpha=0.12, linewidth=0, color="tab:green")

    ax.scatter(
        [goal[0]], [goal[1]], [100],
        marker="*", s=180, label="goal"
    )

    line, = ax.plot([], [], [], linewidth=2)
    aircraft_tri, = ax.plot([], [], [], color="tab:blue", linewidth=2)

    world_size = max(5000.0, np.max(traj[:, :2]) + 100.0)
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_zlim(0, max(700, traj[:, 2].max() + 100))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend()

    headings = np.zeros(len(traj), dtype=float)
    for i in range(1, len(traj)):
        dx = traj[i, 0] - traj[i - 1, 0]
        dy = traj[i, 1] - traj[i - 1, 1]
        if np.hypot(dx, dy) > 1e-9:
            headings[i] = math.atan2(dy, dx)
        else:
            headings[i] = headings[i - 1]

    def update(frame):
        p = traj[:frame + 1]
        line.set_data(p[:, 0], p[:, 1])
        line.set_3d_properties(p[:, 2])

        x, y, z = p[-1]
        theta = headings[min(frame, len(headings) - 1)]
        length = 50.0
        width = 20.0
        forward = np.array([math.cos(theta), math.sin(theta), 0.0])
        lateral = np.array([-math.sin(theta), math.cos(theta), 0.0])

        apex = np.array([x, y, z]) + forward * length
        left = np.array([x, y, z]) - forward * length * 0.35 + lateral * width
        right = np.array([x, y, z]) - forward * length * 0.35 - lateral * width

        tri_x = [apex[0], left[0], right[0], apex[0]]
        tri_y = [apex[1], left[1], right[1], apex[1]]
        tri_z = [apex[2], left[2], right[2], apex[2]]
        aircraft_tri.set_data_3d(tri_x, tri_y, tri_z)

        ax.set_title(
            f"Aircraft trajectory  step={frame} / {len(traj)-1}"
        )
        return line, aircraft_tri

    animation = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        interval=40,
        blit=False,
        repeat=False,
    )

    if save_path:
        animation.save(save_path, writer="pillow", fps=20)
        print(f"saved: {save_path}")

    plt.show()


def animate(env: AircraftObstacleEnv, save_path: str | None = None):
    traj = env.get_trajectory()
    obstacles = env.get_obstacle_array()
    goal = env.goal.copy()
    threshold = env.goal_thresholds[env.goal_threshold_index]
    _draw_animation(traj, obstacles, goal, threshold, save_path)


def animate_state_log(log_path: str | Path, save_path: str | None = None):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traj = np.asarray(data["trajectory"], dtype=float)
    obstacles = np.asarray(data["obstacles"], dtype=float)
    goal = np.asarray(data["goal"], dtype=float)
    threshold = float(data.get("goal_threshold", 1000.0))
    _draw_animation(traj, obstacles, goal, threshold, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--state-log", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if args.state_log is not None:
        animate_state_log(args.state_log, args.save)
        return

    if args.checkpoint is None:
        raise SystemExit("Either --state-log or --checkpoint is required.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ray.init(ignore_reinit_error=True)

    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    env = run_episode(algo, args.seed)
    print("event:", env.last_event)
    print("steps:", env.step_count)
    print("obstacles:", len(env.obstacles))
    print("threshold:", env.goal_thresholds[env.goal_threshold_index])

    animate(env, args.save)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
