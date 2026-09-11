# 中文阅读入口：这是 验证 的核心实现，负责“hash”相关职责。 主要入口：`sha256_stream`、`canonical_json_sha256`、`EmbeddedJsonHashResult`、`identify_embedded_json_hash`。
"""Hash utilities with explicit file and embedded-JSON semantics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_stream(path: Path, block_size: int = 65536) -> str | None:
    """Streaming SHA-256 hash of a file.

    Args:
        path: File path to hash.
        block_size: Read block size in bytes (default 64KB).

    Returns:
        Hex digest string, or None if file not found.
    """
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash deterministic JSON bytes without an embedded self-hash field."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EmbeddedJsonHashResult:
    status: str
    semantic: str
    stored: str | None
    computed: str | None


def identify_embedded_json_hash(
    value: Mapping[str, Any],
    *,
    field: str,
) -> EmbeddedJsonHashResult:
    """Identify and verify supported embedded-hash conventions.

    The hash field is always removed before recomputation. This avoids the
    circular and impossible comparison between a JSON file and a digest stored
    inside that same file. The legacy pretty-JSON form remains supported for
    already-frozen candidate manifests.
    """
    unsigned = dict(value)
    stored_value = unsigned.pop(field, None)
    if stored_value is None:
        return EmbeddedJsonHashResult(
            status="SKIPPED",
            semantic="missing_embedded_hash",
            stored=None,
            computed=None,
        )
    stored = str(stored_value).lower()
    canonical = canonical_json_sha256(unsigned)
    if stored == canonical:
        return EmbeddedJsonHashResult(
            status="PASS",
            semantic="canonical_json_excluding_self",
            stored=stored,
            computed=canonical,
        )
    legacy_payload = json.dumps(unsigned, indent=2, default=str).encode("utf-8")
    legacy = hashlib.sha256(legacy_payload).hexdigest()
    if stored == legacy:
        return EmbeddedJsonHashResult(
            status="PASS",
            semantic="legacy_pretty_json_excluding_self",
            stored=stored,
            computed=legacy,
        )
    return EmbeddedJsonHashResult(
        status="FAIL",
        semantic="unrecognized_or_tampered_embedded_hash",
        stored=stored,
        computed=canonical,
    )
