from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from torch.utils.tensorboard import SummaryWriter

from aircraft_env import AircraftObstacleEnv


def evaluate_episode(algo, seed: int = 12345):
    env = AircraftObstacleEnv()
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    done = False

    while not done:
        # RLlib 2.58 keeps compute_single_action for compatibility, although
        # the new RLModule API is preferred for new code.
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "reward": total_reward,
        "steps": info["step"],
        "event": info["event"],
        "trajectory": env.get_trajectory(),
        "goal": env.goal.copy(),
        "obstacles": env.get_obstacle_array(),
        "goal_threshold": float(env.goal_thresholds[env.goal_threshold_index]),
        "goal_threshold_index": int(env.goal_threshold_index),
        "num_obstacles": int(env.num_obstacles),
    }


def save_state_log(result, path: Path):
    payload = {
        "event": result["event"],
        "steps": int(result["steps"]),
        "reward": float(result["reward"]),
        "goal_threshold": float(result["goal_threshold"]),
        "goal_threshold_index": int(result["goal_threshold_index"]),
        "num_obstacles": int(result["num_obstacles"]),
        "goal": np.asarray(result["goal"], dtype=float).tolist(),
        "trajectory": np.asarray(result["trajectory"], dtype=float).tolist(),
        "obstacles": np.asarray(result["obstacles"], dtype=float).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_trajectory_png(result, path: Path, title: str):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    traj = result["trajectory"]
    obstacles = result["obstacles"]
    goal = result["goal"]

    for x, y, r in obstacles:
        # A finite visual cylinder is enough to represent an infinite obstacle.
        z = np.linspace(0, 600, 20)
        theta = np.linspace(0, 2 * np.pi, 24)
        zz, tt = np.meshgrid(z, theta)
        xx = x + r * np.cos(tt)
        yy = y + r * np.sin(tt)
        ax.plot_surface(xx, yy, zz, alpha=0.10, linewidth=0)

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2)
    ax.scatter(
        [traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
        marker="o", s=60, label="start"
    )
    ax.scatter(
        [goal[0]], [goal[1]], [100.0],
        marker="*", s=150, label="goal"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--gpus-per-learner", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path("runs/aircraft_ppo")
    checkpoint_dir = run_dir / "checkpoints"
    figure_dir = run_dir / "trajectories"
    tensorboard_dir = run_dir / "tensorboard"
    state_log_dir = run_dir / "state_logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    state_log_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    ray.init(ignore_reinit_error=True)

    env_config = {
        "world_size": 5000.0,
        "num_obstacles": 0,
        "max_obstacles": 5,
        "max_steps": 600,
    }

    model_config = DefaultModelConfig(
        fcnet_hiddens=[256, 256],
        fcnet_activation="tanh",
        head_fcnet_hiddens=[],
    )

    config = (
        PPOConfig()
        # RLlib 2.58 defaults to the new API stack. This project still uses the
        # legacy Policy-based action API (`compute_single_action()`), so we must
        # explicitly opt back into the old stack for compatibility.
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            AircraftObstacleEnv,
            env_config=env_config,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.gpus_per_learner,
        )
        .rl_module(model_config=model_config)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size_per_learner=4000,
            minibatch_size=256,
            num_epochs=10,
        )
        .debugging(seed=args.seed, log_level="ERROR")
    )

    algo = config.build_algo()

    history_file = run_dir / "training_history.csv"
    with history_file.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "iteration",
            "episode_reward_mean",
            "episode_len_mean",
            "timesteps_total",
            "eval_reward",
            "eval_steps",
            "eval_event",
        ])

        for iteration in range(1, args.iterations + 1):
            result = algo.train()

            # RLlib metric names may differ slightly between API versions.
            reward_mean = result.get("env_runners", {}).get(
                "episode_return_mean",
                result.get("episode_reward_mean", float("nan"))
            )
            len_mean = result.get("env_runners", {}).get(
                "episode_len_mean",
                result.get("episode_len_mean", float("nan"))
            )
            timesteps = result.get(
                "num_env_steps_sampled_lifetime",
                result.get("timesteps_total", 0)
            )

            eval_reward = ""
            eval_steps = ""
            eval_event = ""

            if iteration % args.eval_every == 0 or iteration == 1:
                evaluation = evaluate_episode(algo, seed=10000 + iteration)
                eval_reward = evaluation["reward"]
                eval_steps = evaluation["steps"]
                eval_event = evaluation["event"]

                save_trajectory_png(
                    evaluation,
                    figure_dir / f"iter_{iteration:05d}.png",
                    f"Evaluation trajectory - iteration {iteration} - {evaluation['event']}",
                )
                save_state_log(
                    {**evaluation, "iteration": iteration},
                    state_log_dir / f"iter_{iteration:05d}.json",
                )

                checkpoint_path = checkpoint_dir / f"iter_{iteration:05d}"
                algo.save(checkpoint_path)

            csv_writer.writerow([
                iteration,
                reward_mean,
                len_mean,
                timesteps,
                eval_reward,
                eval_steps,
                eval_event,
            ])
            f.flush()

            current_goal_threshold = AircraftObstacleEnv.goal_thresholds[AircraftObstacleEnv.goal_threshold_index]
            current_num_obstacles = AircraftObstacleEnv.num_obstacles
            print(
                f"iter={iteration:4d} "
                f"reward={reward_mean!s:>10} "
                f"len={len_mean!s:>8} "
                f"eval={eval_reward!s:>10} "
                f"goal_threshold={current_goal_threshold:>5.0f}m "
                f"num_obstacles={current_num_obstacles} "
                f"event={eval_event}"
            )

            tb_writer.add_scalar("train/reward_mean", float(reward_mean), iteration)
            tb_writer.add_scalar("train/episode_len_mean", float(len_mean), iteration)
            tb_writer.add_scalar("train/timesteps_total", int(timesteps), iteration)
            if eval_reward != "":
                tb_writer.add_scalar("eval/reward", float(eval_reward), iteration)
                tb_writer.add_scalar("eval/steps", int(eval_steps), iteration)
            tb_writer.flush()

    tb_writer.close()
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
