"""DEFProvider — LLMProvider using the DEF Transformer (200M) as language chip.

Drop-in replacement for OllamaProvider. The RAPA-OS kernel sees no difference —
AgentDLLM works unchanged. Only the LLM backend switches from Ollama to DEF.

The DEF model is NOT instruction-tuned. It generates story-like continuations.
The kernel's deterministic hint injection (agent_d_llm.py lines 84-95) ensures
correct behavior regardless of output quality. The model is the "chip", not the "brain".
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

Message = Dict[str, str]


@dataclass
class DEFProvider:
    """LLMProvider that uses the DEF transformer for text generation.

    Satisfies the LLMProvider Protocol (structural subtyping):
        def chat(self, messages, *, temperature, max_tokens) -> str

    Usage:
        from def_transformer.model.config import DEFTransformerConfig
        from def_transformer.model.utils import build_model

        config = DEFTransformerConfig.from_yaml("configs/def_200m_e2_kv.yaml")
        model = build_model(config).eval().to(device)
        ckpt = torch.load("best_model.pt", map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

        llm = DEFProvider(model=model, tokenizer=tokenizer, device=device)
        agent_d = AgentDLLM(llm)  # Drop-in!
    """
    model: torch.nn.Module
    tokenizer: object  # GPT2TokenizerFast
    device: torch.device = torch.device("cpu")
    canvas_manager: object = None  # Optional KernelCanvasManager
    _call_count: int = field(default=0, repr=False)

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> str:
        """Generate text completion from message list.

        Concatenates system + user messages into a single prompt,
        optionally prepends canvas memory, then generates autoregressively.
        """
        self._call_count += 1

        # Build prompt from messages
        prompt = ""
        if self.canvas_manager is not None:
            prefix = self.canvas_manager.to_prefix()
            if prefix:
                prompt = prefix + "\n"
        for msg in messages:
            prompt += msg.get("content", "") + "\n"

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(self.device)

        # Autoregressive generation (no KV-cache for simplicity)
        generated = self._generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        # Decode only the generated part (not the prompt)
        new_tokens = generated[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        """Simple autoregressive generation with early stopping.

        Stops at:
        - EOS token
        - Double newline (\\n\\n) — prevents babble loop
        - "TAGS:" line followed by newline — output complete
        - max_new_tokens reached
        """
        eos_id = self.tokenizer.eos_token_id or 50256
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_id[0] if newline_id else None

        ids = input_ids.clone()
        consecutive_newlines = 0
        saw_tags = False

        for _ in range(max_new_tokens):
            # Forward pass (full sequence, no KV-cache)
            logits, _ = self.model(ids)
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Temperature sampling
            if temperature > 0:
                probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)
            tok_id = next_token.item()

            # Stop conditions
            if tok_id == eos_id:
                break

            # Double newline detection (babble-loop prevention)
            if newline_id is not None and tok_id == newline_id:
                consecutive_newlines += 1
                if consecutive_newlines >= 2:
                    break
            else:
                consecutive_newlines = 0

            # TAGS: line detection (output complete)
            if saw_tags and tok_id == newline_id:
                break
            decoded_so_far = self.tokenizer.decode(
                ids[0, input_ids.shape[1]:].tolist()[-10:])
            if "TAGS:" in decoded_so_far:
                saw_tags = True

        return ids
