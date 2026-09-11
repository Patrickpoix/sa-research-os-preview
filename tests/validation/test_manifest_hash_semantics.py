# 中文阅读入口：这是 验证 的回归测试，验证“清单 / hash / semantics”相关契约；fixture 与断言只用于证明行为，不是生产逻辑的第二 owner。 主要入口：`test_canonical_hash_excludes_self_reference`、`test_legacy_pretty_hash_is_identified_without_file_self_hash`、`test_missing_embedded_hash_is_skipped_not_failed`。
from __future__ import annotations

import hashlib
import json

from alpha_research_os.validation.hash import (
    canonical_json_sha256,
    identify_embedded_json_hash,
)


def test_canonical_hash_excludes_self_reference() -> None:
    unsigned = {"schema": "v1", "rows": [1, 2, 3]}
    payload = dict(unsigned, manifest_sha256=canonical_json_sha256(unsigned))

    result = identify_embedded_json_hash(payload, field="manifest_sha256")

    assert result.status == "PASS"
    assert result.semantic == "canonical_json_excluding_self"
    assert result.computed == payload["manifest_sha256"]


def test_legacy_pretty_hash_is_identified_without_file_self_hash() -> None:
    unsigned = {"schema": "v1", "rows": [1, 2, 3]}
    legacy_bytes = json.dumps(unsigned, indent=2, default=str).encode("utf-8")
    payload = dict(unsigned, manifest_sha256=hashlib.sha256(legacy_bytes).hexdigest())

    result = identify_embedded_json_hash(payload, field="manifest_sha256")

    assert result.status == "PASS"
    assert result.semantic == "legacy_pretty_json_excluding_self"


def test_missing_embedded_hash_is_skipped_not_failed() -> None:
    result = identify_embedded_json_hash({"schema": "v1"}, field="manifest_sha256")

    assert result.status == "SKIPPED"
    assert result.semantic == "missing_embedded_hash"
