"""Logging: human-readable console + file log, structured JSONL events."""

from __future__ import annotations

import logging
import os
import sys

from .common import append_jsonl, utc_now

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger("jevlab")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(ch)
    root.addHandler(fh)
    return root


class EventLog:
    def __init__(self, path: str):
        self.path = path

    def __call__(self, kind: str, **data) -> None:
        append_jsonl(self.path, {"ts": utc_now(), "event": kind, **data})
