"""Multi-env PPO test using RAPA-N's SharedEncoder — 8 parallel envs.

Same architecture as test_ppo_minimal.py but with vectorized environment
collection, matching the vanilla PPO baseline's data collection strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from typing import List

from env.doorkey import DoorKeyEnv
from models.rapa_n_nets import (
    SharedEncoder, SACActorNet5, PPOValueNet5,
    ACTION_NAMES, N_ACTIONS,
)


def run_multienv_ppo(
    size: int = 6,
    seed: int = 42,
    total_steps: int = 500_000,
    max_steps: int = 200,
    n_envs: int = 8,
    n_steps_per_env: int = 128,
):
    encoder = SharedEncoder()
    actor = SACActorNet5()
    critic = PPOValueNet5()

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(actor.parameters())
        + list(critic.parameters()),
        lr=2.5e-4,
        eps=1e-5,
    )

    # PPO hyperparams (matching vanilla PPO baseline exactly)
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    n_epochs = 4
    n_minibatches = 4
    rollout_size = n_envs * n_steps_per_env  # 1024

    param_count = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in actor.parameters())
        + sum(p.numel() for p in critic.parameters())
    )
    print(f"Params: {param_count:,}")
    print(f"Rollout: {n_envs} envs x {n_steps_per_env} steps = {rollout_size}")

    # Create environments
    envs = []
    obs_list = []
    ep_seeds = list(range(seed, seed + 10000))
    env_ep_idx = [0] * n_envs  # episode counter per env
    for i in range(n_envs):
        env = DoorKeyEnv(size=size, seed=ep_seeds[i * 100], max_steps=max_steps)
        obs = env.reset()
        envs.append(env)
        obs_list.append(obs)
        env_ep_idx[i] = i * 100

    total_env_steps = 0
    n_updates = 0
    recent_sr = deque(maxlen=100)
    ep_rewards = [0.0] * n_envs

    while total_env_steps < total_steps:
        # Collect rollout from all envs
        # Store RAW OBS (not embeddings) so we can recompute with gradients
        buf_obs = []  # raw image tensors (3, 7, 7)
        buf_actions = []
        buf_log_probs = []
        buf_values = []
        buf_rewards = []
        buf_dones = []

        for step in range(n_steps_per_env):
            # Batch encode all observations
            imgs = []
            for obs in obs_list:
                img = torch.from_numpy(
                    obs.image.astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                imgs.append(img)
            img_batch = torch.stack(imgs)  # (n_envs, 3, 7, 7)

            with torch.no_grad():
                emb_batch = encoder(img_batch)  # (n_envs, 64)
                logits = actor(emb_batch)       # (n_envs, 5)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                actions = dist.sample()          # (n_envs,)
                log_probs = dist.log_prob(actions)  # (n_envs,)
                values = critic(emb_batch).squeeze(-1)  # (n_envs,)

            # Step all environments
            for i in range(n_envs):
                action_name = ACTION_NAMES[actions[i].item()]
                obs_next, reward, done = envs[i].step(action_name)
                ep_rewards[i] += reward
                total_env_steps += 1

                buf_obs.append(img_batch[i])  # store raw image, NOT embedding
                buf_actions.append(actions[i].item())
                buf_log_probs.append(log_probs[i].item())
                buf_values.append(values[i].item())
                buf_rewards.append(reward - 0.001)  # small step penalty
                buf_dones.append(done)

                if done:
                    success = ep_rewards[i] > 0
                    recent_sr.append(1.0 if success else 0.0)
                    ep_rewards[i] = 0.0

                    # Reset environment with new seed
                    env_ep_idx[i] += 1
                    envs[i].close()
                    envs[i] = DoorKeyEnv(
                        size=size,
                        seed=ep_seeds[env_ep_idx[i] % len(ep_seeds)],
                        max_steps=max_steps,
                    )
                    obs_list[i] = envs[i].reset()
                else:
                    obs_list[i] = obs_next

        # Compute bootstrap values for last step
        imgs = []
        for obs in obs_list:
            img = torch.from_numpy(
                obs.image.astype(np.float32) / 255.0
            ).permute(2, 0, 1)
            imgs.append(img)
        img_batch = torch.stack(imgs)
        with torch.no_grad():
            last_emb = encoder(img_batch)
            last_values = critic(last_emb).squeeze(-1)

        # Compute GAE for the flat rollout buffer
        n_buf = len(buf_obs)
        advantages = [0.0] * n_buf
        returns = [0.0] * n_buf

        # Process GAE per environment (interleaved in buffer)
        for env_i in range(n_envs):
            gae = 0.0
            # Indices for this env in the flat buffer
            env_indices = list(range(env_i, n_buf, n_envs))

            for j in reversed(range(len(env_indices))):
                idx = env_indices[j]
                if j == len(env_indices) - 1:
                    next_val = last_values[env_i].item()
                else:
                    next_idx = env_indices[j + 1]
                    next_val = buf_values[next_idx]

                delta = (buf_rewards[idx]
                         + gamma * next_val * (1.0 - float(buf_dones[idx]))
                         - buf_values[idx])
                gae = (delta
                       + gamma * gae_lambda * (1.0 - float(buf_dones[idx])) * gae)
                advantages[idx] = gae
                returns[idx] = gae + buf_values[idx]

        # PPO update — recompute embeddings WITH gradients (end-to-end)
        obs_t = torch.stack(buf_obs)  # (n_buf, 3, 7, 7)
        act_t = torch.tensor(buf_actions, dtype=torch.long)
        old_lp_t = torch.tensor(buf_log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        mb_size = max(1, n_buf // n_minibatches)
        for _ in range(n_epochs):
            indices = torch.randperm(n_buf)
            for start in range(0, n_buf, mb_size):
                end = min(start + mb_size, n_buf)
                mb = indices[start:end]

                # CRITICAL: Recompute embeddings WITH gradients
                # This allows end-to-end training of encoder + actor + critic
                mb_emb = encoder(obs_t[mb])  # (mb_size, 64) WITH grad_fn
                logits = actor(mb_emb)
                dist = Categorical(F.softmax(logits, dim=-1))
                new_lp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_t[mb]).exp()
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()

                vals = critic(mb_emb).squeeze(-1)
                value_loss = F.mse_loss(vals, ret_t[mb])

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(actor.parameters())
                    + list(critic.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

        n_updates += 1

        # Logging
        sr = sum(recent_sr) / len(recent_sr) if recent_sr else 0.0
        if n_updates % 10 == 0 or total_env_steps >= total_steps:
            print(f"update {n_updates:4d}  "
                  f"env_steps={total_env_steps:>7,}  "
                  f"SR={sr:.0%}  "
                  f"episodes={len(recent_sr)}")

    # Cleanup
    for env in envs:
        env.close()

    final_sr = sum(recent_sr) / len(recent_sr) if recent_sr else 0.0
    print(f"\n=== Multi-Env PPO Result ===")
    print(f"SR: {final_sr:.0%}")
    print(f"Total env steps: {total_env_steps:,}")
    print(f"PPO updates: {n_updates}")
    print(f"Params: {param_count:,}")


if __name__ == "__main__":
    run_multienv_ppo()
