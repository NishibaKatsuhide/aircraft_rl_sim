from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


class AircraftObstacleEnv(gym.Env):
    """
    3D aircraft obstacle-avoidance environment.

    The policy outputs an absolute desired heading.
    The low-level flight-control model converts that command into position changes.
    """

    metadata = {"render_modes": ["human"]}

    MAX_OBSTACLES = 12
    goal_thresholds = [1000.0, 800.0, 600.0, 400.0, 200.0]
    goal_threshold_index = 0
    num_obstacles = 0

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        self.world_size = float(config.get("world_size", 5000.0))
        self.max_altitude = float(config.get("max_altitude", 1000.0))
        self.dt = float(config.get("dt", 1.0))

        self.speed = float(config.get("speed", 100.0))              # m/s
        self.max_turn_rate = math.radians(float(config.get("max_turn_deg_s", 10.0)))
        self.cruise_altitude = float(config.get("cruise_altitude", 300.0))
        self.max_climb_rate = float(config.get("max_climb_rate", 10.0))
        self.max_descent_rate = float(config.get("max_descent_rate", 5.0))

        self.aircraft_radius = float(config.get("aircraft_radius", 10.0))
        self.goal_radius = float(config.get("goal_radius", 75.0))
        self.goal_altitude_tolerance = float(config.get("goal_altitude_tolerance", 50.0))
        self.goal_thresholds = self.__class__.goal_thresholds
        self.goal_threshold_index = self.__class__.goal_threshold_index
        self.num_obstacles = self.__class__.num_obstacles

        self.min_obstacle_radius = float(config.get("min_obstacle_radius", 60.0))
        self.max_obstacle_radius = float(config.get("max_obstacle_radius", 180.0))
        self.max_obstacles = int(config.get("max_obstacles", 5))
        self.num_obstacles = min(
            int(config.get("num_obstacles", 0)),
            self.max_obstacles,
        )

        self.max_steps = int(config.get("max_steps", 600))
        self.render_mode = config.get("render_mode")

        # [aircraft x,y,z, goal x,y, 12 * (obstacle x,y,active)]
        self.obs_dim = 3 + 2 + self.MAX_OBSTACLES * 3

        low = np.full(self.obs_dim, -1.0, dtype=np.float32)
        high = np.full(self.obs_dim, 1.0, dtype=np.float32)

        # Active flags are [0, 1], while all physical coordinates are normalized.
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Absolute desired heading, radians.
        self.action_space = spaces.Box(
            low=np.array([-math.pi], dtype=np.float32),
            high=np.array([math.pi], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng()
        self.aircraft = np.zeros(3, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.heading = 0.0
        self.obstacles: list[Obstacle] = []
        self.step_count = 0
        self.trajectory: list[np.ndarray] = []
        self.last_event = ""

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.last_event = ""
        self.goal_threshold_index = self.__class__.goal_threshold_index
        self.num_obstacles = self.__class__.num_obstacles

        # Start near the lower-left area; goal near the upper-right area.
        self.aircraft = np.array(
            [
                self.rng.uniform(0.05, 0.25) * self.world_size,
                self.rng.uniform(0.05, 0.90) * self.world_size,
                self.rng.uniform(80.0, 150.0),
            ],
            dtype=np.float64,
        )

        self.goal = np.array(
            [
                self.rng.uniform(0.75, 0.95) * self.world_size,
                self.rng.uniform(0.10, 0.90) * self.world_size,
            ],
            dtype=np.float64,
        )

        direct_heading = math.atan2(
            self.goal[1] - self.aircraft[1],
            self.goal[0] - self.aircraft[0],
        )
        self.heading = direct_heading + self.rng.normal(0.0, math.radians(10.0))

        self.obstacles = self._generate_obstacles(self.num_obstacles)

        self.trajectory = [self.aircraft.copy()]
        return self._get_obs(), self._info()

    def step(self, action):
        desired_heading = float(np.asarray(action).reshape(-1)[0])
        desired_heading = self._wrap_angle(desired_heading)

        # Low-level heading controller:
        # aircraft turns toward commanded heading, limited by max turn rate.
        heading_error = self._wrap_angle(desired_heading - self.heading)
        turn = np.clip(heading_error, -self.max_turn_rate * self.dt,
                        self.max_turn_rate * self.dt)
        self.heading = self._wrap_angle(self.heading + turn)

        old_position = self.aircraft.copy()

        # Horizontal motion.
        dx = self.speed * self.dt * math.cos(self.heading)
        dy = self.speed * self.dt * math.sin(self.heading)
        self.aircraft[0] += dx
        self.aircraft[1] += dy

        boundary_hit = (
            self.aircraft[0] <= 0.0 or self.aircraft[0] >= self.world_size or
            self.aircraft[1] <= 0.0 or self.aircraft[1] >= self.world_size
        )

        # Keep the aircraft within the world bounds, where x/y are constrained to
        # the same 0..world_size domain as the goal and obstacles.
        self.aircraft[0] = float(np.clip(self.aircraft[0], 0.0, self.world_size))
        self.aircraft[1] = float(np.clip(self.aircraft[1], 0.0, self.world_size))

        # Simple vertical autopilot:
        # climb to cruise altitude, then descend near the goal.
        horizontal_goal_distance = np.linalg.norm(self.aircraft[:2] - self.goal)
        if horizontal_goal_distance < 600.0:
            target_altitude = 100.0
        else:
            target_altitude = self.cruise_altitude

        altitude_error = target_altitude - self.aircraft[2]
        if altitude_error > 0:
            dz = min(altitude_error, self.max_climb_rate * self.dt)
        else:
            dz = max(altitude_error, -self.max_descent_rate * self.dt)

        self.aircraft[2] = float(np.clip(
            self.aircraft[2] + dz, 0.0, self.max_altitude
        ))

        self.step_count += 1
        self.trajectory.append(self.aircraft.copy())

        collision = self._collision()
        goal = self._goal_reached()

        # Per-step time penalty encourages shorter trajectories.
        reward = -1.0

        terminated = False
        truncated = False

        if boundary_hit:
            reward = -500.0
            terminated = True
            self.last_event = "boundary"
        elif collision:
            reward = -1000.0
            terminated = True
            self.last_event = "collision"
        elif goal:
            # Faster arrival = larger reward.
            reward = 1000.0 - 2.0 * self.step_count
            terminated = True
            self.last_event = "goal"
        elif self.step_count >= self.max_steps:
            # Safety cutoff for learning. It is a truncation, not a terminal
            # success/collision event.
            truncated = True
            self.last_event = "timeout"

        # Tiny progress shaping makes early learning less sparse.
        # It does not dominate the terminal rewards.
        if not terminated:
            old_dist = np.linalg.norm(old_position[:2] - self.goal)
            new_dist = np.linalg.norm(self.aircraft[:2] - self.goal)
            reward += float(np.clip((old_dist - new_dist) / 50.0, -1.0, 1.0))

        return self._get_obs(), reward, terminated, truncated, self._info()

    # ----------------------------
    # Environment internals
    # ----------------------------

    def _generate_obstacles(self, count: int):
        obstacles = []
        attempts = 0
        while len(obstacles) < count and attempts < 5000:
            attempts += 1
            x = self.rng.uniform(0.05, 0.95) * self.world_size
            y = self.rng.uniform(0.05, 0.95) * self.world_size
            radius = self.rng.uniform(
                self.min_obstacle_radius, self.max_obstacle_radius
            )

            if np.linalg.norm(np.array([x, y]) - self.aircraft[:2]) < radius + 250:
                continue
            if np.linalg.norm(np.array([x, y]) - self.goal) < radius + 250:
                continue

            if any(
                math.hypot(x - o.x, y - o.y) < radius + o.radius + 80
                for o in obstacles
            ):
                continue

            obstacles.append(Obstacle(x, y, radius))
        return obstacles

    def _collision(self) -> bool:
        for obstacle in self.obstacles:
            d = math.hypot(
                self.aircraft[0] - obstacle.x,
                self.aircraft[1] - obstacle.y,
            )
            if d <= obstacle.radius + self.aircraft_radius:
                return True
        return False

    def _goal_reached(self) -> bool:
        horizontal_distance = np.linalg.norm(self.aircraft[:2] - self.goal)
        target_threshold = self.goal_thresholds[self.goal_threshold_index]
        reached = horizontal_distance <= target_threshold

        if reached:
            if self.goal_threshold_index < len(self.goal_thresholds) - 1:
                self.goal_threshold_index += 1
            else:
                self.goal_threshold_index = 0
                if self.num_obstacles < self.max_obstacles:
                    self.num_obstacles += 1

            self.__class__.goal_threshold_index = self.goal_threshold_index
            self.__class__.num_obstacles = self.num_obstacles

        return reached

    def _get_obs(self):
        half_world = self.world_size / 2.0

        values = [
            (self.aircraft[0] - half_world) / half_world,
            (self.aircraft[1] - half_world) / half_world,
            self.aircraft[2] / self.max_altitude,
            (self.goal[0] - half_world) / half_world,
            (self.goal[1] - half_world) / half_world,
        ]

        for i in range(self.MAX_OBSTACLES):
            if i < len(self.obstacles):
                o = self.obstacles[i]
                values.extend([
                    (o.x - half_world) / half_world,
                    (o.y - half_world) / half_world,
                    1.0,
                ])
            else:
                values.extend([0.0, 0.0, 0.0])

        values = np.asarray(values, dtype=np.float32)
        return np.clip(values, -1.0, 1.0)

    def _info(self):
        return {
            "step": self.step_count,
            "event": self.last_event,
            "aircraft_xyz": self.aircraft.copy(),
            "goal_xy": self.goal.copy(),
            "heading_rad": self.heading,
            "heading_deg": math.degrees(self.heading),
            "num_obstacles": len(self.obstacles),
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def get_trajectory(self):
        return np.asarray(self.trajectory)

    def get_obstacle_array(self):
        return np.asarray([[o.x, o.y, o.radius] for o in self.obstacles])

    def render(self):
        # Visualization is intentionally handled by visualize.py.
        return {
            "aircraft": self.aircraft.copy(),
            "goal": self.goal.copy(),
            "obstacles": self.get_obstacle_array(),
        }
