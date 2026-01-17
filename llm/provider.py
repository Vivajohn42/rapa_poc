from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Protocol
import requests

Message = Dict[str, str]  # {"role": "...", "content": "..."}


class LLMProvider(Protocol):
    def chat(self, messages: List[Message], *, temperature: float = 0.2, max_tokens: int = 256) -> str:
        ...


@dataclass
class OllamaProvider:
    """
    Ollama native chat endpoint: http://localhost:11434/api/chat
    """
    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    timeout_s: int = 120

    def chat(self, messages: List[Message], *, temperature: float = 0.2, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),  # Ollama token cap
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "")
