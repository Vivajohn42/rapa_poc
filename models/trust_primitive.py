"""TrustPrimitive: stream-agnostic trust signal from logits.

Computes trust = clamp(w_margin * margin_norm + w_entropy * (1 - entropy_norm), 0, 1)
where:
  margin = logit_top1 - logit_top2  (logit space, pre-softmax)
  entropy_norm = H(softmax(logits)) / log(n_classes)  [0..1]
  margin_norm = clamp(margin / margin_scale, 0, 1)

Why this is better than softmax max-prob:
  - Softmax max-prob for 4 classes: floor=0.25, trained range=0.35-0.65 (compressed)
  - Margin in logit space: floor=0, no upper bound, much better dynamic range
  - Entropy: uniform=1.0 (trust=0), peaked=~0.1 (trust~0.9)
  - Combined signal separates "sure" from "guessing" far more clearly

Reusable for any classifier with n_classes >= 2.
"""
from __future__ import annotations

import math
from typing import Dict

import torch


class TrustPrimitive:
    """Compute a calibrated trust scalar from raw logits.

    Args:
        n_classes: Number of output classes (default: 4 for direction net).
        w_margin: Weight for margin component (default: 0.6).
        w_entropy: Weight for entropy component (default: 0.4).
        margin_scale: Margin normalization scale. Margins are divided by this
            value before clamping to [0,1]. Controls sensitivity — higher
            scale means you need a larger margin to reach trust=1.
            Default: 3.0 (a logit margin of 3 maps to margin_norm=1.0).
    """

    def __init__(
        self,
        n_classes: int = 4,
        w_margin: float = 0.6,
        w_entropy: float = 0.4,
        margin_scale: float = 3.0,
    ):
        self.n_classes = n_classes
        self.w_margin = w_margin
        self.w_entropy = w_entropy
        self.margin_scale = margin_scale
        self._max_entropy = math.log(n_classes)  # ln(4) ≈ 1.386

    def compute_trust(self, logits: torch.Tensor) -> float:
        """Compute trust from a single logit vector.

        Args:
            logits: (n_classes,) raw logits tensor.

        Returns:
            trust: float in [0, 1]. Higher = more trustworthy.
        """
        # Top-2 logit margin
        top2 = torch.topk(logits, k=min(2, self.n_classes))
        margin = (top2.values[0] - top2.values[1]).item()
        margin_norm = max(0.0, min(1.0, margin / self.margin_scale))

        # Entropy of softmax distribution
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum().item()
        entropy_norm = entropy / self._max_entropy  # [0, 1]

        # Combined trust
        trust = self.w_margin * margin_norm + self.w_entropy * (1.0 - entropy_norm)
        return max(0.0, min(1.0, trust))

    def compute_trust_batch(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute trust for a batch of logit vectors.

        Args:
            logits: (batch, n_classes) raw logits tensor.

        Returns:
            trust: (batch,) tensor of trust values in [0, 1].
        """
        # Top-2 margin
        top2 = torch.topk(logits, k=min(2, self.n_classes), dim=-1)
        margins = top2.values[:, 0] - top2.values[:, 1]  # (batch,)
        margin_norm = (margins / self.margin_scale).clamp(0.0, 1.0)

        # Entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)  # (batch,)
        entropy_norm = entropy / self._max_entropy

        # Combined trust
        trust = self.w_margin * margin_norm + self.w_entropy * (1.0 - entropy_norm)
        return trust.clamp(0.0, 1.0)

    def decompose(self, logits: torch.Tensor) -> Dict[str, float]:
        """Return all components for diagnostics.

        Returns dict with: margin, margin_norm, entropy, entropy_norm, trust,
        softmax_conf (old metric for comparison).
        """
        top2 = torch.topk(logits, k=min(2, self.n_classes))
        margin = (top2.values[0] - top2.values[1]).item()
        margin_norm = max(0.0, min(1.0, margin / self.margin_scale))

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum().item()
        entropy_norm = entropy / self._max_entropy

        trust = self.w_margin * margin_norm + self.w_entropy * (1.0 - entropy_norm)
        trust = max(0.0, min(1.0, trust))

        return {
            "margin": margin,
            "margin_norm": margin_norm,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "trust": trust,
            "softmax_conf": probs.max().item(),
        }
