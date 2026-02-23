"""MiniGrid DoorKey wrappers for PPO baselines.

Two observation modes:
  1. Raw 7x7x3 partial obs (no privileged state, no features) → DoorKeyGymWrapper
  2. 60-dim feature vector (same features as RAPA's Stream C) → DoorKeyFeaturesWrapper

DoorKeyGymWrapper / DoorKeyVecEnv: raw pixel observations
DoorKeyFeaturesWrapper / DoorKeyFeaturesVecEnv: engineered feature observations
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

import minigrid  # noqa: F401  — registers MiniGrid envs

# MiniGrid object-type codes (from gen_obs image channel 0)
_OBJ_UNSEEN = 0
_OBJ_EMPTY = 1
_OBJ_WALL = 2
_OBJ_FLOOR = 3
_OBJ_DOOR = 4
_OBJ_KEY = 5
_OBJ_GOAL = 8

# Door state codes (image channel 2 for door objects)
_DOOR_OPEN = 0

# Direction vectors: 0=right, 1=down, 2=left, 3=up
_DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class DoorKeyGymWrapper:
    """Raw MiniGrid observations for monolithic PPO baseline.

    Returns:
      obs: (7, 7, 3) float32 normalized [0, 1]  (MiniGrid partial ego-view)
      reward: float (sparse: ~0.9 at goal, 0 otherwise)
      done: bool

    Action space: 7 discrete MiniGrid actions (0-6).
    PPO must learn which actions are useful from experience.
    """

    def __init__(
        self,
        size: int = 6,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        self._seed = seed
        self._max_steps = max_steps or (10 * size * size)
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        self._env = gym.make(env_id, max_steps=self._max_steps)
        self.n_actions = 7  # Full MiniGrid action space
        self.obs_shape = (7, 7, 3)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment. Returns (7, 7, 3) float32 observation."""
        obs, _info = self._env.reset(seed=seed or self._seed)
        return obs["image"].astype(np.float32) / 255.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action. Returns (obs, reward, done)."""
        obs, reward, terminated, truncated, _info = self._env.step(action)
        done = terminated or truncated
        return obs["image"].astype(np.float32) / 255.0, float(reward), done

    def close(self):
        self._env.close()


class DoorKeyVecEnv:
    """Simple vectorized wrapper for N parallel MiniGrid DoorKey environments.

    Auto-resets environments on done. Designed for PPO rollout collection.
    """

    def __init__(
        self,
        size: int = 6,
        n_envs: int = 8,
        seed_base: int = 0,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        self.n_envs = n_envs
        self._max_steps = max_steps or (10 * size * size)
        self._seed_base = seed_base
        self.n_actions = 7
        self.obs_shape = (7, 7, 3)

        # Create N environments with different seeds
        self._envs: List[gym.Env] = []
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        for i in range(n_envs):
            self._envs.append(
                gym.make(env_id, max_steps=self._max_steps))

        # Episode tracking for metrics
        self._episode_steps = np.zeros(n_envs, dtype=np.int32)
        self._episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self._episode_count = 0
        self._completed_episodes: List[Tuple[bool, int, float]] = []
        # (success, steps, reward)

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (n_envs, 7, 7, 3) float32."""
        obs_list = []
        for i, env in enumerate(self._envs):
            seed = self._seed_base + self._episode_count + i
            obs, _ = env.reset(seed=seed)
            obs_list.append(obs["image"].astype(np.float32) / 255.0)
            self._episode_steps[i] = 0
            self._episode_rewards[i] = 0.0
        return np.stack(obs_list)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all environments. Auto-resets on done.

        Returns: (obs, rewards, dones) — all (n_envs,) or (n_envs, 7, 7, 3).
        """
        obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)

        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, _info = env.step(
                int(actions[i]))
            done = terminated or truncated
            rewards[i] = float(reward)
            dones[i] = done

            self._episode_steps[i] += 1
            self._episode_rewards[i] += float(reward)

            if done:
                # Record completed episode
                success = reward > 0
                self._completed_episodes.append((
                    success,
                    int(self._episode_steps[i]),
                    float(self._episode_rewards[i]),
                ))
                self._episode_count += 1

                # Auto-reset
                seed = self._seed_base + self._episode_count * 100 + i
                obs, _ = env.reset(seed=seed)
                self._episode_steps[i] = 0
                self._episode_rewards[i] = 0.0

            obs_list.append(obs["image"].astype(np.float32) / 255.0)

        return np.stack(obs_list), rewards, dones

    def get_completed_episodes(self) -> List[Tuple[bool, int, float]]:
        """Drain completed episode buffer. Returns [(success, steps, reward)]."""
        eps = list(self._completed_episodes)
        self._completed_episodes.clear()
        return eps

    def close(self):
        for env in self._envs:
            env.close()


# ── Feature-based wrappers (same 60-dim features as RAPA) ──────────

def _ego_to_world(
    ego_i: int, ego_j: int,
    agent_pos: Tuple[int, int], agent_dir: int,
) -> Optional[Tuple[int, int]]:
    """Convert MiniGrid ego-view (i,j) to world (x,y).

    Matches ObjectMemory._ego_to_world exactly.
    """
    rel_i = ego_i - 3
    rel_j = 6 - ego_j
    if agent_dir == 0:      # facing right
        wx, wy = rel_j, rel_i
    elif agent_dir == 1:    # facing down
        wx, wy = -rel_i, rel_j
    elif agent_dir == 2:    # facing left
        wx, wy = -rel_j, -rel_i
    elif agent_dir == 3:    # facing up
        wx, wy = rel_i, -rel_j
    else:
        return None
    return (agent_pos[0] + wx, agent_pos[1] + wy)


class _MiniObjectMemory:
    """Lightweight object memory for feature extraction.

    Replicates ObjectMemory's ego-view parsing without importing the
    full RAPA agent stack.  Used by DoorKeyFeaturesWrapper to provide
    the same 60-dim features that RAPA's Stream C uses.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.known_walls: Set[Tuple[int, int]] = set()
        self.known_empty: Set[Tuple[int, int]] = set()
        self.visited: Set[Tuple[int, int]] = set()
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.door_state: Optional[int] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.carrying_key: bool = False
        self._key_picked_up: bool = False

    def reset(self):
        self.known_walls.clear()
        self.known_empty.clear()
        self.visited.clear()
        self.key_pos = None
        self.door_pos = None
        self.door_state = None
        self.goal_pos = None
        self.carrying_key = False
        self._key_picked_up = False

    def update(self, env_unwrapped) -> None:
        """Parse the 7x7 ego-view — same logic as ObjectMemory.update()."""
        obs = env_unwrapped.gen_obs()
        image = obs["image"]
        agent_dir = int(env_unwrapped.agent_dir)
        agent_pos = (int(env_unwrapped.agent_pos[0]),
                     int(env_unwrapped.agent_pos[1]))

        self.visited.add(agent_pos)
        self.known_empty.add(agent_pos)

        self.carrying_key = env_unwrapped.carrying is not None
        if self.carrying_key:
            self._key_picked_up = True
            self.key_pos = None

        for ei in range(7):
            for ej in range(7):
                obj_type = int(image[ei, ej, 0])
                obj_state = int(image[ei, ej, 2])
                if obj_type == _OBJ_UNSEEN:
                    continue
                wp = _ego_to_world(ei, ej, agent_pos, agent_dir)
                if wp is None:
                    continue
                wx, wy = wp
                if not (0 <= wx < self.grid_size
                        and 0 <= wy < self.grid_size):
                    continue

                if obj_type == _OBJ_WALL:
                    self.known_walls.add(wp)
                    self.known_empty.discard(wp)
                elif obj_type in (_OBJ_EMPTY, _OBJ_FLOOR):
                    self.known_empty.add(wp)
                elif obj_type == _OBJ_KEY:
                    if not self._key_picked_up:
                        self.key_pos = wp
                    self.known_empty.add(wp)
                elif obj_type == _OBJ_DOOR:
                    self.door_pos = wp
                    self.door_state = obj_state
                    if obj_state == _DOOR_OPEN:
                        self.known_empty.add(wp)
                    else:
                        self.known_empty.discard(wp)
                elif obj_type == _OBJ_GOAL:
                    self.goal_pos = wp
                    self.known_empty.add(wp)

    @property
    def door_open(self) -> bool:
        return self.door_state == _DOOR_OPEN

    @property
    def obstacles(self) -> Set[Tuple[int, int]]:
        obs = set(self.known_walls)
        if (self.door_pos is not None
                and self.door_state is not None
                and self.door_state != _DOOR_OPEN):
            obs.add(self.door_pos)
        return obs

    @property
    def phase(self) -> str:
        if self.carrying_key:
            return "reach_goal" if self.door_open else "open_door"
        return "find_key"

    @property
    def phase_int(self) -> int:
        """0=find_key, 1=open_door, 2=reach_goal."""
        p = self.phase
        if p == "find_key":
            return 0
        elif p == "open_door":
            return 1
        return 2

    @property
    def target(self) -> Tuple[int, int]:
        """Current navigation target based on phase.

        Same logic as AutonomousDoorKeyAgentC._pick_target().
        """
        p = self.phase
        if p == "find_key" and self.key_pos is not None:
            return self.key_pos
        if p == "open_door" and self.door_pos is not None:
            return self.door_pos
        if p == "reach_goal" and self.goal_pos is not None:
            return self.goal_pos
        # Fallback: center of grid (exploration heuristic)
        c = self.grid_size // 2
        return (c, c)


def _extract_features(
    agent_pos: Tuple[int, int],
    agent_dir: int,
    mem: _MiniObjectMemory,
) -> np.ndarray:
    """Build 60-dim feature vector identical to RAPA's extract_online_features().

    Layout:
      [0:2]   relative target delta (normalized)
      [2:51]  7x7 local obstacle window (49)
      [51:55] direction one-hot (4)
      [55:58] phase one-hot (3)
      [58]    carrying_key flag
      [59]    door_open flag
    """
    size = mem.grid_size
    norm = float(max(size, 1))
    target = mem.target
    obstacles = mem.obstacles

    features = np.zeros(60, dtype=np.float32)

    # [0:2] Relative target delta
    features[0] = (target[0] - agent_pos[0]) / norm
    features[1] = (target[1] - agent_pos[1]) / norm

    # [2:51] 7x7 local obstacle window
    ax, ay = agent_pos
    half = 3
    obs_set = obstacles
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            wx, wy = ax + dx, ay + dy
            idx = 2 + (dy + half) * 7 + (dx + half)
            if not (0 <= wx < size and 0 <= wy < size):
                features[idx] = 1.0  # boundary = obstacle
            elif (wx, wy) in obs_set:
                features[idx] = 1.0

    # [51:55] Direction one-hot
    features[51 + (agent_dir % 4)] = 1.0

    # [55:58] Phase one-hot
    features[55 + mem.phase_int] = 1.0

    # [58] carrying_key
    features[58] = 1.0 if mem.carrying_key else 0.0

    # [59] door_open
    features[59] = 1.0 if mem.door_open else 0.0

    return features


class DoorKeyFeaturesWrapper:
    """60-dim feature observations for PPO+Features baseline.

    Returns the SAME features that RAPA's Stream C (SACAgentC) uses,
    allowing a fair comparison that isolates architecture from features.

    obs: (60,) float32
    reward: float (sparse: ~0.9 at goal, 0 otherwise)
    done: bool

    Action space: 7 discrete MiniGrid actions (same as vanilla PPO).
    """

    def __init__(
        self,
        size: int = 6,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        self._seed = seed
        self._max_steps = max_steps or (10 * size * size)
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        self._env = gym.make(env_id, max_steps=self._max_steps)
        self.n_actions = 7
        self.obs_shape = (60,)
        self._mem = _MiniObjectMemory(grid_size=size)

    def _get_features(self) -> np.ndarray:
        uw = self._env.unwrapped
        self._mem.update(uw)
        agent_pos = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))
        agent_dir = int(uw.agent_dir)
        return _extract_features(agent_pos, agent_dir, self._mem)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment. Returns (60,) float32 feature vector."""
        self._env.reset(seed=seed or self._seed)
        self._mem.reset()
        return self._get_features()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action. Returns (features, reward, done)."""
        _obs, reward, terminated, truncated, _info = self._env.step(action)
        done = terminated or truncated
        return self._get_features(), float(reward), done

    def close(self):
        self._env.close()


class DoorKeyFeaturesVecEnv:
    """Vectorized 60-dim feature env for PPO+Features training.

    Same interface as DoorKeyVecEnv but returns (n_envs, 60) features.
    """

    def __init__(
        self,
        size: int = 6,
        n_envs: int = 8,
        seed_base: int = 0,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        self.n_envs = n_envs
        self._max_steps = max_steps or (10 * size * size)
        self._seed_base = seed_base
        self.n_actions = 7
        self.obs_shape = (60,)

        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        self._envs: List[gym.Env] = []
        self._mems: List[_MiniObjectMemory] = []
        for _ in range(n_envs):
            self._envs.append(gym.make(env_id, max_steps=self._max_steps))
            self._mems.append(_MiniObjectMemory(grid_size=size))

        self._episode_steps = np.zeros(n_envs, dtype=np.int32)
        self._episode_rewards = np.zeros(n_envs, dtype=np.float32)
        self._episode_count = 0
        self._completed_episodes: List[Tuple[bool, int, float]] = []

    def _get_features(self, i: int) -> np.ndarray:
        uw = self._envs[i].unwrapped
        self._mems[i].update(uw)
        agent_pos = (int(uw.agent_pos[0]), int(uw.agent_pos[1]))
        agent_dir = int(uw.agent_dir)
        return _extract_features(agent_pos, agent_dir, self._mems[i])

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns (n_envs, 60) float32."""
        obs_list = []
        for i, env in enumerate(self._envs):
            seed = self._seed_base + self._episode_count + i
            env.reset(seed=seed)
            self._mems[i].reset()
            obs_list.append(self._get_features(i))
            self._episode_steps[i] = 0
            self._episode_rewards[i] = 0.0
        return np.stack(obs_list)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all environments. Auto-resets on done.

        Returns: (features, rewards, dones).
        """
        obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)

        for i, env in enumerate(self._envs):
            _obs, reward, terminated, truncated, _info = env.step(
                int(actions[i]))
            done = terminated or truncated
            rewards[i] = float(reward)
            dones[i] = done

            self._episode_steps[i] += 1
            self._episode_rewards[i] += float(reward)

            if done:
                success = reward > 0
                self._completed_episodes.append((
                    success,
                    int(self._episode_steps[i]),
                    float(self._episode_rewards[i]),
                ))
                self._episode_count += 1

                seed = self._seed_base + self._episode_count * 100 + i
                env.reset(seed=seed)
                self._mems[i].reset()
                self._episode_steps[i] = 0
                self._episode_rewards[i] = 0.0

            obs_list.append(self._get_features(i))

        return np.stack(obs_list), rewards, dones

    def get_completed_episodes(self) -> List[Tuple[bool, int, float]]:
        """Drain completed episode buffer."""
        eps = list(self._completed_episodes)
        self._completed_episodes.clear()
        return eps

    def close(self):
        for env in self._envs:
            env.close()
