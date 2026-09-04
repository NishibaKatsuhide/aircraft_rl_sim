from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ray

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


def animate(env: AircraftObstacleEnv, save_path: str | None = None):
    traj = env.get_trajectory()
    obstacles = env.get_obstacle_array()
    goal = env.goal.copy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Show each infinite cylinder as a finite visual cylinder.
    for x, y, r in obstacles:
        z = np.linspace(0, 600, 16)
        theta = np.linspace(0, 2 * np.pi, 24)
        zz, tt = np.meshgrid(z, theta)
        xx = x + r * np.cos(tt)
        yy = y + r * np.sin(tt)
        ax.plot_surface(xx, yy, zz, alpha=0.10, linewidth=0)

    ax.scatter(
        [goal[0]], [goal[1]], [100],
        marker="*", s=180, label="goal"
    )

    line, = ax.plot([], [], [], linewidth=2)
    aircraft_point, = ax.plot([], [], [], marker="o", markersize=8)

    ax.set_xlim(traj[:, 0].min() - 300, traj[:, 0].max() + 300)
    ax.set_ylim(traj[:, 1].min() - 300, traj[:, 1].max() + 300)
    ax.set_zlim(0, max(700, traj[:, 2].max() + 100))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend()

    def update(frame):
        p = traj[:frame + 1]
        line.set_data(p[:, 0], p[:, 1])
        line.set_3d_properties(p[:, 2])

        aircraft_point.set_data([p[-1, 0]], [p[-1, 1]])
        aircraft_point.set_3d_properties([p[-1, 2]])

        ax.set_title(
            f"Aircraft trajectory  step={frame} / {len(traj)-1}"
        )
        return line, aircraft_point

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Restore a trained RLlib Algorithm.
    algo = ray.rllib.algorithms.algorithm.Algorithm.from_checkpoint(
        args.checkpoint
    )

    env = run_episode(algo, args.seed)
    print("event:", env.last_event)
    print("steps:", env.step_count)

    animate(env, args.save)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
