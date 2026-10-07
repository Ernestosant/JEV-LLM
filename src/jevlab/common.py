"""Small shared helpers: IO, hashing, time, config hashing."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(obj) -> str:
    return sha256_text(canonical_json(obj))


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, record: dict) -> None:
    """Append + flush + fsync: records survive a VM disconnect."""
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def write_json(path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def subprocess_env() -> dict:
    """Environment for `python -m jevlab.*` subprocesses: make the package importable."""
    src = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def load_prompt(path) -> str:
    """Prompt files keep a trailing newline on disk; the protocol text has none."""
    return Path(path).read_text(encoding="utf-8").strip("\n")
