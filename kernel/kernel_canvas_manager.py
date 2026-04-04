"""KernelCanvasManager — text-level working memory controlled by the kernel.

Replaces the neural Canvas (KVWriteHead/KVCanvasAttention) with a simple
Python dict. The kernel decides WHAT to store and WHEN — no learned write gate.

Facts are serialized as a text prefix prepended to the DEF model's prompt,
making stored knowledge explicitly visible in the language model's context.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional


class KernelCanvasManager:
    """Kernel-controlled text-level working memory.

    Conceptually maps to 8 Canvas slots, but stored as a Python dict.
    The kernel writes facts explicitly — no gradient, no training needed.
    """

    def __init__(self, n_slots: int = 8) -> None:
        self.n_slots = n_slots
        self.slots: OrderedDict[str, str] = OrderedDict()

    def write(self, key: str, value: str) -> None:
        """Write a fact to a named slot. Evicts oldest if full."""
        if key in self.slots:
            # Update existing slot (move to end = most recent)
            del self.slots[key]
        elif len(self.slots) >= self.n_slots:
            # Evict oldest slot
            self.slots.popitem(last=False)
        self.slots[key] = value

    def read(self, key: str) -> Optional[str]:
        """Read a specific slot by key."""
        return self.slots.get(key)

    def to_prefix(self) -> str:
        """Serialize all slots as a text block for prompt injection."""
        if not self.slots:
            return ""
        lines = ["[MEMORY]"]
        for key, value in self.slots.items():
            lines.append(f"{key}: {value}")
        lines.append("[/MEMORY]")
        return "\n".join(lines)

    def update_from_observation(self, obs) -> None:
        """Extract facts from environment observation.

        Accepts either a ZA object or a raw GridState/dict from the environment.
        Called by the demo script after each kernel tick.
        """
        # Handle both ZA (pydantic) and raw GridState/dict
        if hasattr(obs, "agent_pos"):
            self.write("agent_pos", str(obs.agent_pos))
        elif isinstance(obs, dict) and "agent_pos" in obs:
            self.write("agent_pos", str(obs["agent_pos"]))

        # Capture hints (critical for GridWorld)
        hint = getattr(obs, "hint", None)
        if hint is None and isinstance(obs, dict):
            hint = obs.get("hint")
        if hint:
            self.write(f"hint_{hint}", f"Hint {hint} discovered")

    def update_from_narrative(self, zD) -> None:
        """Extract facts from D's output (ZD schema)."""
        if not hasattr(zD, "meaning_tags"):
            return
        for tag in zD.meaning_tags:
            if tag.startswith("hint:"):
                hint_id = tag.split(":")[1].strip().upper()
                self.write(f"hint_{hint_id}", f"Hint {hint_id} -> goal location")
            elif tag.startswith("target:"):
                target = tag.split(":")[1].strip()
                self.write("target", f"Target at {target}")

    def reset(self) -> None:
        """Clear all slots for new episode."""
        self.slots.clear()

    def __repr__(self) -> str:
        return f"KernelCanvasManager({len(self.slots)}/{self.n_slots} slots)"
