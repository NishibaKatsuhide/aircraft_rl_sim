from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="runs/aircraft_ppo/training_history.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.plot(df["iteration"], df["episode_reward_mean"], label="train reward")
    if df["eval_reward"].notna().any():
        ax.plot(df["iteration"], df["eval_reward"], "o-", label="evaluation reward")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Aircraft RL training progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = Path(args.csv).with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
