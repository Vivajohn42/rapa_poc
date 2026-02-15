"""
Utilities for LLM-based evaluation tests (Stufe 7).

Provides:
- Ollama availability checks (graceful skip when Ollama is not running)
- Model discovery (which models are locally available)
- Timed LLM calls for latency measurement
"""

import time
from typing import List, Tuple, Optional

import requests

from llm.provider import OllamaProvider

DEFAULT_MODELS = [
    "phi3:mini",        # 3.8B — first as smoke test
    "mistral:latest",   # 7B — proven baseline
    "qwen2.5:3b",       # 3B — mid-size
    "gemma2:2b",        # 2B — most risky, last
]


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running by hitting GET /api/tags."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Return list of locally installed model names."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        return []


def check_model_available(model: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if a specific model is pulled locally."""
    available = list_available_models(base_url)
    # Ollama may list as "phi3:mini" or "phi3:mini:latest" etc.
    for m in available:
        if m == model or m.startswith(model.split(":")[0] + ":"):
            # Exact match or prefix match on model family
            if model in m or m == model:
                return True
    # More lenient: check if the base name matches
    model_base = model.split(":")[0]
    return any(m.split(":")[0] == model_base for m in available)


def timed_llm_call(
    provider: OllamaProvider,
    messages: list,
    **kwargs,
) -> Tuple[str, float]:
    """
    Wrapper that measures LLM response time.

    Returns:
        (response_text, latency_ms)
    """
    start = time.perf_counter()
    result = provider.chat(messages, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def get_available_test_models(
    requested: Optional[List[str]] = None,
    base_url: str = "http://localhost:11434",
) -> List[str]:
    """
    Filter requested models to only those available locally.

    Args:
        requested: Specific models to check. If None, uses DEFAULT_MODELS.
        base_url: Ollama base URL.

    Returns:
        List of model names that are locally available, in request order.
    """
    if requested is None:
        requested = DEFAULT_MODELS

    available = list_available_models(base_url)
    if not available:
        return []

    result = []
    for model in requested:
        model_base = model.split(":")[0]
        for avail in available:
            avail_base = avail.split(":")[0]
            if model_base == avail_base:
                # Use the exact name from Ollama's list
                result.append(model)
                break

    return result
