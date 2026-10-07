"""Model snapshots pinned by SHA, file-hash verification, parameter inventory, token sets."""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from pathlib import Path

from .common import sha256_file

log = logging.getLogger("jevlab.models")

WEIGHT_SUFFIXES = (".safetensors",)
META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "jevk5_config.json", "model.safetensors.index.json",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
              "vocab.json", "merges.txt", "SHA256SUMS")


def snapshot(repo: str, revision: str, cache_dir: str | None = None) -> str:
    """Download (or reuse) the snapshot at an exact commit. Never uses a mutable branch."""
    from huggingface_hub import snapshot_download

    if len(revision) != 40:
        raise ValueError(f"revision must be a full commit SHA, got {revision!r}")
    t0 = time.time()
    path = snapshot_download(repo, revision=revision, cache_dir=cache_dir,
                             allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt",
                                             "SHA256SUMS", "LICENSE", "README.md"])
    log.info("snapshot %s@%s -> %s (%.0f s)", repo, revision[:10], path, time.time() - t0)
    return path


def remote_file_hashes(repo: str, revision: str) -> dict[str, dict]:
    """sha256 (LFS) or git blob id per file, as published on the Hub for that commit."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    out = {}
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        sha = None
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[s.rfilename] = {"size": s.size, "sha256": sha, "blob_id": getattr(s, "blob_id", None)}
    return {"commit": info.sha, "files": out}


def verify_snapshot(repo: str, revision: str, path: str, hash_weights: bool = True) -> dict:
    """Hash every local file; compare LFS files with the Hub sha256 and, when the repo
    publishes SHA256SUMS (JevK5), with that list too."""
    remote = remote_file_hashes(repo, revision)
    rep = {"repo": repo, "revision": revision, "remote_commit": remote["commit"],
           "commit_matches": remote["commit"] == revision, "files": {}, "mismatches": []}
    sums = {}
    p_sums = Path(path) / "SHA256SUMS"
    if p_sums.exists():
        for line in p_sums.read_text().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                sums[name.strip()] = h
    for f in sorted(Path(path).rglob("*")):
        if not f.is_file():
            continue
        name = str(f.relative_to(path)).replace("\\", "/")
        is_weight = name.endswith(WEIGHT_SUFFIXES)
        if is_weight and not hash_weights:
            rep["files"][name] = {"size": f.stat().st_size, "sha256": None, "skipped": True}
            continue
        t0 = time.time()
        h = sha256_file(f)
        r = remote["files"].get(name, {})
        ok_lfs = (r.get("sha256") is None) or (r.get("sha256") == h)
        ok_sums = (name not in sums) or (sums[name] == h)
        rep["files"][name] = {"size": f.stat().st_size, "sha256": h, "remote_sha256": r.get("sha256"),
                              "sha256sums": sums.get(name), "ok": ok_lfs and ok_sums,
                              "hash_s": round(time.time() - t0, 2)}
        if not (ok_lfs and ok_sums):
            rep["mismatches"].append(name)
    rep["ok"] = rep["commit_matches"] and not rep["mismatches"]
    return rep


# ----------------------------------------------------------------------- parameter inventory
def _safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def is_language_key(k: str) -> bool:
    kl = k.lower()
    return not any(x in kl for x in ("visual", "vision", "mtp", "merger", "patch_embed"))


def param_inventory(path: str, tie_word_embeddings: bool) -> dict:
    """Count language-model parameters from safetensors headers (embeddings + head included;
    tied weights counted once; vision tower and MTP excluded)."""
    total_lang, total_all, excluded = 0, 0, 0
    by_dtype: dict[str, int] = {}
    keys = []
    for f in sorted(Path(path).glob("*.safetensors")):
        hdr = _safetensors_header(str(f))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            numel = 1
            for d in v["shape"]:
                numel *= d
            total_all += numel
            if is_language_key(k):
                total_lang += numel
                by_dtype[v["dtype"]] = by_dtype.get(v["dtype"], 0) + numel * DTYPE_BYTES.get(v["dtype"], 0)
                keys.append(k)
            else:
                excluded += numel
    has_head = any(k.endswith("lm_head.weight") for k in keys)
    return {"language_params": total_lang, "all_params_in_files": total_all,
            "excluded_params_vision_mtp": excluded, "language_bytes_by_dtype": by_dtype,
            "tie_word_embeddings": tie_word_embeddings, "lm_head_in_files": has_head,
            "n_language_tensors": len(keys)}


# ----------------------------------------------------------------------- token sets
def special_ids(tok) -> dict:
    ids = {}
    for name in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        t = tok.convert_tokens_to_ids(name)
        ids[name] = t
    return {"eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id, "named": ids}


def eos_ids(tok) -> list[int]:
    out = []
    for name in ("<|im_end|>", "<|endoftext|>"):
        t = tok.convert_tokens_to_ids(name)
        if isinstance(t, int) and t >= 0 and t not in out:
            out.append(t)
    if tok.eos_token_id is not None and tok.eos_token_id not in out:
        out.append(tok.eos_token_id)
    return out


def newline_token_ids(tok) -> set[int]:
    """All vocabulary ids whose byte-level text contains a newline (0x0A). Built from the
    raw decoded bytes of each single token, not from pretty-printed token strings."""
    vocab = tok.get_vocab()
    special = set(tok.all_special_ids)
    out = set()
    for t, i in vocab.items():
        if i in special:
            continue
        s = tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "\n" in s:
            out.add(i)
    return out
