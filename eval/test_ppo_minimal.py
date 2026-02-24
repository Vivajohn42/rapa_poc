"""Minimal PPO test using RAPA-N's SharedEncoder — NO kernel, NO streams.

Purpose: Isolate whether the encoder + PPO setup can learn DoorKey at all.
If this works (~50%+ SR), the problem is in the kernel integration.
If this also fails (~10% SR), the problem is in the encoder/PPO architecture.
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
from typing import List, Tuple

from env.doorkey import DoorKeyEnv
from models.rapa_n_nets import SharedEncoder, SACActorNet5, PPOValueNet5, ACTION_NAMES, N_ACTIONS


def run_minimal_ppo(
    size: int = 6,
    seed: int = 42,
    n_episodes: int = 1500,
    max_steps: int = 200,
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

    # PPO hyperparams
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    n_epochs = 4
    n_minibatches = 4
    min_rollout_steps = 1024

    # Rollout buffer (accumulated across episodes)
    buf_embs: List[torch.Tensor] = []
    buf_actions: List[int] = []
    buf_log_probs: List[float] = []
    buf_values: List[float] = []
    buf_rewards: List[float] = []
    buf_dones: List[bool] = []

    total_env_steps = 0
    recent_sr = deque(maxlen=20)
    n_updates = 0

    param_count = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in actor.parameters())
        + sum(p.numel() for p in critic.parameters())
    )
    print(f"Params: {param_count:,}")

    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = DoorKeyEnv(size=size, seed=ep_seed, max_steps=max_steps)
        obs = env.reset()
        done = False
        ep_reward = 0.0

        for t in range(max_steps):
            # Encode observation
            img = torch.from_numpy(
                obs.image.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                emb = encoder(img).squeeze(0)
                logits = actor(emb.unsqueeze(0)).squeeze(0)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_idx)).item()
                value = critic(emb.unsqueeze(0)).squeeze().item()

            action = ACTION_NAMES[action_idx]

            # Step
            obs_next, reward, done = env.step(action)
            ep_reward += reward
            total_env_steps += 1

            # Exploration bonus (same as RAPA-N eval loop)
            # Actually, let's test WITHOUT exploration bonus first
            shaped_reward = reward - 0.001  # small step penalty only

            buf_embs.append(emb)
            buf_actions.append(action_idx)
            buf_log_probs.append(log_prob)
            buf_values.append(value)
            buf_rewards.append(shaped_reward)
            buf_dones.append(done)

            obs = obs_next
            if done:
                break

        env.close()
        success = done and ep_reward > 0
        recent_sr.append(1.0 if success else 0.0)

        # PPO update when we have enough data
        n_buf = len(buf_embs)
        if n_buf >= min_rollout_steps:
            # Compute GAE
            if buf_dones[-1]:
                next_value = 0.0
            else:
                with torch.no_grad():
                    next_value = critic(buf_embs[-1].unsqueeze(0)).squeeze().item()

            advantages = [0.0] * n_buf
            returns = [0.0] * n_buf
            gae = 0.0
            for i in reversed(range(n_buf)):
                if i == n_buf - 1:
                    nv = next_value
                else:
                    nv = buf_values[i + 1]
                delta = buf_rewards[i] + gamma * nv * (1.0 - float(buf_dones[i])) - buf_values[i]
                gae = delta + gamma * gae_lambda * (1.0 - float(buf_dones[i])) * gae
                advantages[i] = gae
                returns[i] = gae + buf_values[i]

            emb_batch = torch.stack(buf_embs)
            act_batch = torch.tensor(buf_actions, dtype=torch.long)
            old_lp = torch.tensor(buf_log_probs, dtype=torch.float32)
            adv_batch = torch.tensor(advantages, dtype=torch.float32)
            ret_batch = torch.tensor(returns, dtype=torch.float32)

            if len(adv_batch) > 1:
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

            for _ in range(n_epochs):
                indices = torch.randperm(n_buf)
                mb_size = max(1, n_buf // n_minibatches)
                for start in range(0, n_buf, mb_size):
                    end = min(start + mb_size, n_buf)
                    mb = indices[start:end]

                    logits = actor(emb_batch[mb])
                    dist = Categorical(F.softmax(logits, dim=-1))
                    new_lp = dist.log_prob(act_batch[mb])
                    entropy = dist.entropy().mean()

                    ratio = (new_lp - old_lp[mb]).exp()
                    s1 = ratio * adv_batch[mb]
                    s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch[mb]
                    policy_loss = -torch.min(s1, s2).mean()

                    vals = critic(emb_batch[mb]).squeeze(-1)
                    value_loss = F.mse_loss(vals, ret_batch[mb])

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
            buf_embs.clear()
            buf_actions.clear()
            buf_log_probs.clear()
            buf_values.clear()
            buf_rewards.clear()
            buf_dones.clear()

        # Logging
        if ep < 5 or ep % 20 == 19 or ep == n_episodes - 1:
            sr = sum(recent_sr) / len(recent_sr) if recent_sr else 0.0
            print(f"ep {ep:4d}: {'OK' if success else 'FAIL':4s}  "
                  f"steps={t+1 if done else max_steps:3d}  "
                  f"SR={sr:.0%}  "
                  f"updates={n_updates}  "
                  f"env_steps={total_env_steps:,}")

    final_sr = sum(recent_sr) / len(recent_sr) if recent_sr else 0.0
    print(f"\n=== Minimal PPO Result ({n_episodes} eps) ===")
    print(f"SR: {final_sr:.0%}")
    print(f"Total env steps: {total_env_steps:,}")
    print(f"PPO updates: {n_updates}")
    print(f"Params: {param_count:,}")


if __name__ == "__main__":
    run_minimal_ppo()
