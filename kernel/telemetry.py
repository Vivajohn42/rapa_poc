"""Telemetry subsystem for rapa_mvp kernel.

Provides two first-class observability primitives:
  - MetricRegistry: named counters + gauges, queryable for display/CSV
  - EventSink: append-only JSONL structured event log

Usage:
    tel = Telemetry(log_dir=Path("runs/my_run"))
    tel.metrics.gauge("regime", 3)
    tel.metrics.inc("regime_changes")
    tel.events.emit("regime_change", old="ORIENT", new="EXPLORE")

    # At episode end:
    tel.metrics.reset_episode()  # clear episode-scoped metrics

    # At run end:
    tel.close()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricRegistry:
    """Named counters and gauges, queryable for display and CSV export.

    Counters are monotonically increasing integers.
    Gauges are floating-point values that can be set to any value.

    Episode-scoped metrics (prefixed with "ep.") are reset each episode.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}

    def inc(self, name: str, delta: int = 1) -> None:
        """Increment a counter by delta (default 1)."""
        self._counters[name] = self._counters.get(name, 0) + delta

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value."""
        self._gauges[name] = value

    def get(self, name: str, default: float = 0.0) -> float:
        """Get a metric value (counter or gauge)."""
        if name in self._gauges:
            return self._gauges[name]
        if name in self._counters:
            return float(self._counters[name])
        return default

    def get_counter(self, name: str) -> int:
        """Get a counter value."""
        return self._counters.get(name, 0)

    def snapshot(self) -> Dict[str, float]:
        """Return all metrics as a flat dict."""
        result: Dict[str, float] = {}
        for k, v in self._counters.items():
            result[k] = float(v)
        for k, v in self._gauges.items():
            result[k] = v
        return result

    def reset_episode(self) -> None:
        """Reset episode-scoped metrics (prefixed with 'ep.')."""
        ep_counter_keys = [k for k in self._counters if k.startswith("ep.")]
        for k in ep_counter_keys:
            del self._counters[k]
        ep_gauge_keys = [k for k in self._gauges if k.startswith("ep.")]
        for k in ep_gauge_keys:
            del self._gauges[k]


class EventSink:
    """Append-only JSONL event log.

    Events are buffered in memory and flushed to disk periodically
    or on close(). If no path is given, events are only kept in memory
    (useful for testing or when disk logging is disabled).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._buffer: List[Dict[str, Any]] = []
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")

    def emit(self, event_type: str, **data: Any) -> None:
        """Log a structured event.

        Each event gets a monotonic timestamp and the event type.
        Additional key-value pairs are stored as-is.
        """
        event = {
            "ts": time.monotonic(),
            "type": event_type,
            **data,
        }
        self._buffer.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event, default=str) + "\n")

    def flush(self) -> None:
        """Flush buffered events to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Access in-memory event buffer (read-only)."""
        return self._buffer

    def clear_buffer(self) -> None:
        """Clear in-memory buffer (file is unaffected)."""
        self._buffer.clear()


class Telemetry:
    """Bundles MetricRegistry + EventSink into one object.

    Pass to MvpKernel constructor to enable shadow-mode telemetry.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.metrics = MetricRegistry()
        event_path = log_dir / "events.jsonl" if log_dir is not None else None
        self.events = EventSink(event_path)

    def close(self) -> None:
        """Flush and close all sinks."""
        self.events.flush()
        self.events.close()
