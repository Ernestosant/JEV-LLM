"""Seed derivation (protocol §5.2). Never uses Python's hash()."""

from __future__ import annotations

import hashlib
import random

MOD = 2**63 - 1


def derive(*parts) -> int:
    """First 8 bytes of SHA-256 of the '|'-joined UTF-8 string, little-endian, mod 2**63-1."""
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % MOD


def branch_seed(seed: int, problem_id: str, condition: str, round_index: int,
                branch_index: int, generation: int = 0) -> int:
    """`seed|problem_id|condition|round_index|branch_index|generation`."""
    return derive(seed, problem_id, condition, round_index, branch_index, generation)


def option_order_seed(seed: int, problem_id: str, condition: str, round_index: int,
                      generation: int = 0) -> int:
    """Selector permutation seed: distinct derivation ending in `|option_order`."""
    return derive(seed, problem_id, condition, round_index, generation, "option_order")


def permutation(n: int, perm_seed: int) -> list[int]:
    """perm[pos] = original branch index shown at option position `pos`."""
    order = list(range(n))
    random.Random(perm_seed).shuffle(order)
    return order
