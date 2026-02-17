"""Neural Agent A: perception with learned belief embedding.

Extends the deterministic AgentA by computing a belief embedding via
GridEncoder and attaching it to the ZA state.  All downstream agents
and kernel logic remain unchanged (embedding is Optional).

Usage:
    encoder = GridEncoder()
    encoder.load_state_dict(torch.load("train/checkpoints/grid_encoder.pt"))
    agent_a = NeuralAgentA(encoder)
"""
from __future__ import annotations

import torch
from state.schema import ZA
from kernel.interfaces import StreamA
from models.grid_encoder import GridEncoder, encode_grid_observation


class NeuralAgentA(StreamA):
    """Perception agent that produces ZA with a neural embedding."""

    def __init__(self, encoder: GridEncoder, device: str = "cpu"):
        self.encoder = encoder
        self.device = device
        self.encoder.to(device)
        self.encoder.eval()

    def infer_zA(self, obs) -> ZA:
        """Convert raw observation to ZA with embedding.

        Works with any observation object that has the standard fields:
        width, height, agent_pos, goal_pos, obstacles, hint.
        """
        # Build base ZA fields
        width = obs.width
        height = obs.height
        agent_pos = obs.agent_pos
        goal_pos = obs.goal_pos
        obstacles = list(obs.obstacles)
        hint = getattr(obs, "hint", None)

        # Compute embedding
        obs_tensor = encode_grid_observation(
            width=width,
            height=height,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            hint=hint,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.encoder(obs_tensor).squeeze(0).cpu().tolist()

        return ZA(
            width=width,
            height=height,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            hint=hint,
            embedding=embedding,
        )
