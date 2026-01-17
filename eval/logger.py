import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class JSONLLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(self, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        return obj

    def log(self, record: Dict[str, Any]) -> None:
        rec = self._to_jsonable(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
