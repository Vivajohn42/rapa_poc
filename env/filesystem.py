"""Filesystem Sandbox Environment — real OS calls, read-only.

A directory tree with text files. The agent must find a hidden piece of
information (a "secret") by navigating directories and reading files.
No BFS possible — the agent must reason about file names and contents.

Actions: ls, cd <dir>, read <file>, search <pattern>, answer <text>
Observation: current directory listing + last file content snippet

The environment creates a temporary directory tree with:
  - Multiple subdirectories (logs/, config/, data/, docs/, etc.)
  - Text files with realistic content
  - ONE file somewhere contains the secret
  - Distractor files with plausible but irrelevant content
"""
from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class FileSystemObs:
    """Observation from the filesystem environment."""
    current_dir: str           # relative path from sandbox root
    files: List[str]           # files in current directory
    dirs: List[str]            # subdirectories in current directory
    last_read: str             # last file content (truncated)
    last_read_name: str        # name of last read file
    step: int
    hint: Optional[str] = None  # for RAPA-OS compatibility


# ─── Scenario Templates ───────────────────────────────────────

SCENARIOS = [
    {
        "name": "webserver_debug",
        "goal": "Find why the web server is returning 503 errors",
        "secret": "upstream_timeout",
        "tree": {
            "logs": {
                "access.log": "192.168.1.1 - GET /index.html 200\n192.168.1.2 - GET /api/data 200\n192.168.1.3 - GET /status 200\n",
                "error.log": "ERROR: upstream server timeout after 30s\nERROR: 503 Service Unavailable returned to client\nCAUSE: upstream_timeout\n",
                "auth.log": "LOGIN user=admin status=success\nLOGIN user=deploy status=success\n",
            },
            "config": {
                "nginx.conf": "server { listen 80; proxy_pass http://backend:8080; proxy_timeout 30s; }\n",
                "app.conf": "DATABASE_URL=postgres://localhost/mydb\nDEBUG=false\n",
            },
            "data": {
                "users.csv": "id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com\n",
                "metrics.json": '{"requests": 1523, "errors": 47, "uptime": "23h"}\n',
            },
        },
    },
    {
        "name": "missing_config",
        "goal": "Find the database password that the application needs",
        "secret": "hunter2_prod",
        "tree": {
            "app": {
                "main.py": "import os\nDB_PASS = os.environ.get('DB_PASSWORD')\nif not DB_PASS:\n    raise ValueError('DB_PASSWORD not set')\n",
                "requirements.txt": "flask==2.3\npsycopg2==2.9\nredis==4.5\n",
            },
            "config": {
                "dev.env": "DB_PASSWORD=devpass123\nREDIS_URL=localhost:6379\n",
                "prod.env": "DB_PASSWORD=hunter2_prod\nREDIS_URL=redis-prod:6379\n",
                "README.md": "Config files for each environment.\nprod.env contains production credentials.\n",
            },
            "docs": {
                "setup.md": "# Setup\n1. Copy config/*.env to .env\n2. Run docker-compose up\n",
                "architecture.md": "# Architecture\nFlask app + PostgreSQL + Redis\n",
            },
        },
    },
    {
        "name": "security_audit",
        "goal": "Find which user account has been compromised",
        "secret": "user_jenkins",
        "tree": {
            "logs": {
                "auth.log": "2024-01-15 LOGIN user=admin ip=10.0.0.1 status=OK\n2024-01-15 LOGIN user=jenkins ip=10.0.0.5 status=OK\n2024-01-15 LOGIN user=jenkins ip=203.0.113.99 status=OK\n2024-01-15 LOGIN user=jenkins ip=203.0.113.99 status=OK\n",
                "firewall.log": "ALLOW 10.0.0.0/8 -> any\nBLOCK 203.0.113.0/24 -> port 22\nALERT: 203.0.113.99 bypassed firewall via port 443\n",
                "system.log": "service sshd restarted\nservice nginx running\nservice postgres running\n",
            },
            "users": {
                "accounts.txt": "admin: role=superadmin, last_login=10.0.0.1\njenkins: role=ci, last_login=203.0.113.99\ndeploy: role=deploy, last_login=10.0.0.2\n",
                "permissions.csv": "user,read,write,admin\nadmin,yes,yes,yes\njenkins,yes,yes,no\ndeploy,yes,no,no\n",
            },
            "reports": {
                "weekly.txt": "No incidents reported this week.\nAll systems nominal.\n",
                "alert.txt": "ALERT: Unusual login pattern detected for user_jenkins.\nSource IP 203.0.113.99 is external and blocked by firewall rules.\nThis account may be compromised.\n",
            },
        },
    },
]


class FileSystemEnv:
    """Read-only filesystem sandbox with hidden secrets."""

    def __init__(self, scenario: int = 0, seed: int = 42):
        self.rng = random.Random(seed)
        self.scenario = SCENARIOS[scenario % len(SCENARIOS)]
        self.goal = self.scenario["goal"]
        self.secret = self.scenario["secret"]
        self._sandbox: Optional[Path] = None
        self._current_dir: str = "/"
        self._last_read: str = ""
        self._last_read_name: str = ""
        self._step: int = 0
        self._solved: bool = False
        self.available_actions = ["ls", "cd", "read", "search", "answer"]

    def reset(self) -> FileSystemObs:
        """Create the sandbox directory tree and return initial observation."""
        if self._sandbox and self._sandbox.exists():
            shutil.rmtree(self._sandbox)

        self._sandbox = Path(tempfile.mkdtemp(prefix="rapa_fs_"))
        self._create_tree(self._sandbox, self.scenario["tree"])
        self._current_dir = "/"
        self._last_read = ""
        self._last_read_name = ""
        self._step = 0
        self._solved = False
        return self._observe()

    def step(self, action: str) -> Tuple[FileSystemObs, float, bool]:
        """Execute an action and return (obs, reward, done)."""
        self._step += 1
        reward = 0.0
        done = False

        parts = action.strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "ls":
            pass  # observation already shows listing

        elif cmd == "cd" and arg:
            new_dir = self._resolve_path(arg)
            full = self._sandbox / new_dir.lstrip("/")
            if full.is_dir():
                self._current_dir = new_dir

        elif cmd == "read" and arg:
            fpath = self._sandbox / self._current_dir.lstrip("/") / arg
            if fpath.is_file():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    self._last_read = content[:500]
                    self._last_read_name = arg
                except Exception:
                    self._last_read = "(error reading file)"
                    self._last_read_name = arg

        elif cmd == "search" and arg:
            # Search current directory for files containing pattern
            results = []
            cdir = self._sandbox / self._current_dir.lstrip("/")
            for f in cdir.iterdir():
                if f.is_file():
                    try:
                        if arg.lower() in f.read_text(encoding="utf-8").lower():
                            results.append(f.name)
                    except Exception:
                        pass
            self._last_read = f"Files containing '{arg}': {', '.join(results) if results else 'none'}"
            self._last_read_name = f"search:{arg}"

        elif cmd == "answer" and arg:
            if self.secret.lower() in arg.lower():
                reward = 1.0
                self._solved = True
                done = True
            else:
                reward = -0.1  # wrong answer penalty

        # Timeout
        if self._step >= 30:
            done = True

        return self._observe(), reward, done

    def _observe(self) -> FileSystemObs:
        cdir = self._sandbox / self._current_dir.lstrip("/")
        files = []
        dirs = []
        if cdir.is_dir():
            for item in sorted(cdir.iterdir()):
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    dirs.append(item.name + "/")
        return FileSystemObs(
            current_dir=self._current_dir,
            files=files,
            dirs=dirs,
            last_read=self._last_read,
            last_read_name=self._last_read_name,
            step=self._step,
        )

    def _resolve_path(self, arg: str) -> str:
        if arg == "..":
            parts = self._current_dir.rstrip("/").split("/")
            return "/".join(parts[:-1]) or "/"
        if arg.startswith("/"):
            return arg
        return self._current_dir.rstrip("/") + "/" + arg

    def _create_tree(self, base: Path, tree: dict) -> None:
        for name, content in tree.items():
            path = base / name
            if isinstance(content, dict):
                path.mkdir(parents=True, exist_ok=True)
                self._create_tree(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    def cleanup(self) -> None:
        if self._sandbox and self._sandbox.exists():
            shutil.rmtree(self._sandbox)

    @property
    def goal_positions(self):
        """Compatibility with RAPA-OS kernel."""
        return {"secret": (0, 0)}
