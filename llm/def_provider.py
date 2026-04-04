"""DEFProvider — LLMProvider using the DEF Transformer (200M) as language chip.

Drop-in replacement for OllamaProvider. The RAPA-OS kernel sees no difference —
AgentDLLM works unchanged. Only the LLM backend switches from Ollama to DEF.

The DEF model is NOT instruction-tuned. It generates story-like continuations.
The kernel's deterministic hint injection (agent_d_llm.py lines 84-95) ensures
correct behavior regardless of output quality. The model is the "chip", not the "brain".

Phase F.2: KV-cache for fast autoregressive generation (prefill once, then
           single-token steps). ~3-8x speedup over full-sequence per step.
Phase F.3: Forced prefix injection (e.g. "NARRATIVE:") to guide output format.
Phase F.4: Two-phase generation — narrative + prediction in same KV-cache context.
           Model generates NARRATIVE+TAGS first, then PREDICTION as continuation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

Message = Dict[str, str]


@dataclass
class DEFProvider:
    """LLMProvider that uses the DEF transformer for text generation.

    Satisfies the LLMProvider Protocol (structural subtyping):
        def chat(self, messages, *, temperature, max_tokens) -> str
        def chat(self, messages, *, temperature, max_tokens,
                 prediction_enabled=True) -> tuple[str, str]

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
    forced_prefix: str | None = "NARRATIVE:"  # Injected at start of generation
    _call_count: int = field(default=0, repr=False)
    # KV-cache state from last _generate call (for two-phase continuation)
    _last_past_kv: object = field(default=None, repr=False)
    _last_use_kv_cache: bool = field(default=False, repr=False)

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
        prediction_enabled: bool = False,
    ) -> str | tuple[str, str]:
        """Generate text completion from message list.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Max tokens for narrative + tags.
            prediction_enabled: When True, generates a prediction continuation
                after the narrative and returns (text, prediction).

        Returns:
            str when prediction_enabled=False (backward compatible).
            (str, str) when prediction_enabled=True: (narrative_text, prediction_text).
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

        # Phase 1: Narrative + Tags
        generated_ids = self._generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            forced_prefix=self.forced_prefix,
        )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if not prediction_enabled:
            return text.strip()

        # Phase 2: Prediction (continues from the same KV-cache)
        prediction_ids = self._generate_continuation(
            forced_prefix="\nPREDICTION:",
            max_new_tokens=40,
            temperature=temperature,
        )
        prediction = self.tokenizer.decode(
            prediction_ids, skip_special_tokens=True,
        ).strip()

        return text.strip(), prediction

    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.2,
        forced_prefix: str | None = None,
    ) -> list[int]:
        """Autoregressive generation with KV-cache and early stopping.

        Returns list of generated token ids (not including the prompt).
        Saves KV-cache state for optional continuation via _generate_continuation().

        Stops at:
        - EOS token
        - Double newline (\\n\\n) — prevents babble loop
        - "TAGS:" line followed by newline — output complete
        - max_new_tokens reached
        """
        eos_id = self.tokenizer.eos_token_id or 50256
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_id[0] if newline_id else None

        generated: list[int] = []
        consecutive_newlines = 0
        saw_tags = False

        # Handle forced prefix: append to prompt for prefill
        prefill_ids = input_ids
        prefix_len = 0
        if forced_prefix:
            prefix_token_ids = self.tokenizer.encode(forced_prefix, add_special_tokens=False)
            if prefix_token_ids:
                prefix_len = len(prefix_token_ids)
                prefix_tensor = torch.tensor([prefix_token_ids], device=self.device)
                prefill_ids = torch.cat([input_ids, prefix_tensor], dim=1)
                generated.extend(prefix_token_ids)

        # Try KV-cache path; fall back to no-cache if model doesn't support it
        try:
            result = self.model(prefill_ids, use_cache=True)
            logits, _, past_kv = result
            use_kv_cache = True
        except TypeError:
            logits, _ = self.model(prefill_ids)
            past_kv = None
            use_kv_cache = False

        next_logits = logits[0, -1, :]
        budget = max_new_tokens - prefix_len

        for _ in range(budget):
            if temperature > 0:
                probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            tok_id = next_token.item()
            generated.append(tok_id)

            if tok_id == eos_id:
                break

            if newline_id is not None and tok_id == newline_id:
                consecutive_newlines += 1
                if consecutive_newlines >= 2:
                    break
            else:
                consecutive_newlines = 0

            if saw_tags and tok_id == newline_id:
                break
            decoded_recent = self.tokenizer.decode(generated[-10:])
            if "TAGS:" in decoded_recent:
                saw_tags = True

            if use_kv_cache:
                step_ids = next_token.unsqueeze(0)
                logits, _, past_kv = self.model(
                    step_ids, use_cache=True, past_key_values=past_kv,
                )
            else:
                all_ids = torch.cat(
                    [input_ids, torch.tensor([generated], device=self.device)],
                    dim=1,
                )
                logits, _ = self.model(all_ids)

            next_logits = logits[0, -1, :]

        # Save cache state for optional two-phase continuation
        self._last_past_kv = past_kv
        self._last_use_kv_cache = use_kv_cache
        self._last_input_ids = input_ids
        self._last_generated = generated

        return generated

    @torch.no_grad()
    def _generate_continuation(
        self,
        forced_prefix: str = "\nPREDICTION:",
        max_new_tokens: int = 40,
        temperature: float = 0.2,
    ) -> list[int]:
        """Continue generation from the KV-cache saved by _generate().

        Used for Phase 2 (prediction) after Phase 1 (narrative + tags).
        The model sees its own Phase 1 output as context — enabling coherent
        prediction based on the narrative it just generated.

        Stops at: EOS, double newline, or max_new_tokens.
        """
        if not self._last_use_kv_cache or self._last_past_kv is None:
            # No cache available — can't continue. Return empty.
            return []

        eos_id = self.tokenizer.eos_token_id or 50256
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_id[0] if newline_id else None

        # Inject forced prefix into the cached context
        prefix_token_ids = self.tokenizer.encode(forced_prefix, add_special_tokens=False)
        if not prefix_token_ids:
            return []

        generated: list[int] = list(prefix_token_ids)
        past_kv = self._last_past_kv

        # Feed prefix tokens through the cache one-by-one (cheap: incremental steps)
        for pid in prefix_token_ids:
            step_ids = torch.tensor([[pid]], device=self.device)
            logits, _, past_kv = self.model(
                step_ids, use_cache=True, past_key_values=past_kv,
            )

        next_logits = logits[0, -1, :]
        consecutive_newlines = 0
        budget = max_new_tokens - len(prefix_token_ids)

        for _ in range(budget):
            if temperature > 0:
                probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            tok_id = next_token.item()
            generated.append(tok_id)

            if tok_id == eos_id:
                break

            if newline_id is not None and tok_id == newline_id:
                consecutive_newlines += 1
                if consecutive_newlines >= 2:
                    break
            else:
                consecutive_newlines = 0

            step_ids = next_token.unsqueeze(0)
            logits, _, past_kv = self.model(
                step_ids, use_cache=True, past_key_values=past_kv,
            )
            next_logits = logits[0, -1, :]

        # Strip the forced prefix tokens from the returned ids
        return generated[len(prefix_token_ids):]
