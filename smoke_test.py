import numpy as np
from aircraft_env import AircraftObstacleEnv


def main():
    env = AircraftObstacleEnv()
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs)

    total = 0.0
    for _ in range(10):
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        total += reward
        if terminated or truncated:
            break

    print("OK")
    print("observation dimension:", env.observation_space.shape)
    print("action:", env.action_space)
    print("total reward over smoke test:", total)
    print("event:", info["event"])


if __name__ == "__main__":
    main()
