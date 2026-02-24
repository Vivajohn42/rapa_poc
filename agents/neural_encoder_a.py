"""NeuralEncoderA — Standalone CNN perception stream for RAPA-N.

Processes raw 7×7×3 MiniGrid ego-view observations through a shared CNN
backbone (identical architecture to PPO) and produces a ZA with a 64-dim
learned embedding plus structural fields for kernel governance.

The encoder is NOT trained separately — gradients flow from C's SAC policy
loss backward through the shared backbone (end-to-end, like PPO).
"""
from __future__ import annotations

from typing import Optional

import torch
import numpy as np

from kernel.interfaces import StreamA, StreamLearner, NullLearner
from state.schema import ZA
from models.rapa_n_nets import SharedEncoder


class NeuralEncoderA(StreamA):
    """CNN-based perception stream: raw 7×7×3 obs → ZA with embedding.

    The encoder weights are shared with NeuralSAC_C (passed in constructor).
    A.infer_zA() runs the forward pass; C's training updates the weights.
    """

    def __init__(self, encoder: SharedEncoder):
        self._encoder = encoder
        self._learner_inst = NullLearner(label="encoder-A-passive")

    def infer_zA(self, obs) -> ZA:
        """Convert DoorKeyState observation to ZA with learned embedding.

        obs: DoorKeyState with .image (7,7,3 uint8), .agent_pos, etc.
        """
        # CNN forward pass → 64-dim embedding
        image = obs.image  # (7, 7, 3) uint8
        img_t = torch.from_numpy(image.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 7, 7)

        with torch.no_grad():
            embedding = self._encoder(img_t).squeeze(0)  # (64,)

        return ZA(
            width=obs.width,
            height=obs.height,
            agent_pos=obs.agent_pos,
            goal_pos=obs.goal_pos,
            obstacles=obs.obstacles,
            direction=obs.agent_dir,
            hint=obs.hint,
            embedding=embedding.tolist(),
        )

    @property
    def learner(self) -> StreamLearner:
        return self._learner_inst
