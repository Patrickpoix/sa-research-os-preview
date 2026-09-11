# 中文阅读入口：这是 validator 输出 schema 与 machine-exit 的唯一公共映射。各 domain verifier
# 产生 individual checks，本模块按显式优先级归并为 PASS/FAIL_INTERNAL/DATA_MISSING/NOT_READY/
# BLOCKED_EXTERNAL 等 aggregate status，再映射非零/零退出码。它不重新判断 domain facts；统一
# contract 的目的正是避免“报告写失败但进程 exit 0”或不同脚本各自解释状态。
"""Validation report contract: unified schema, status derivation, exit code.

All validators must use this module instead of writing ad-hoc reports.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

# 单项检查状态：保留“内部失败、外部阻塞、跳过、脚本异常”等差异。
PASS = "PASS"
FAIL = "FAIL"
BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
SKIPPED = "SKIPPED"
SCRIPT_ERROR = "SCRIPT_ERROR"

# 聚合状态：面向 caller/machine，而不是把所有非 PASS 都压成一个 FAIL。
FAIL_INTERNAL = "FAIL_INTERNAL"
DATA_MISSING = "DATA_MISSING"
NOT_READY = "NOT_READY"

VALID_CHECK_STATUSES = {PASS, FAIL, BLOCKED_EXTERNAL, SKIPPED, SCRIPT_ERROR, DATA_MISSING, NOT_READY}


def make_check(
    name: str,
    status: str,
    severity: str = "INFO",
    detail: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a single check result dict."""
    if status not in VALID_CHECK_STATUSES:
        raise ValueError(f"Invalid check status: {status}. Must be one of {VALID_CHECK_STATUSES}")
    return {
        "name": name,
        "status": status,
        "severity": severity,
        "detail": detail,
        "evidence": evidence or {},
    }


def derive_status(checks: list[dict[str, Any]]) -> str:
    """Derive aggregate report status from individual checks.

    Priority order (highest first):
    1. SCRIPT_ERROR → SCRIPT_ERROR
    2. any FAIL → FAIL_INTERNAL
    3. BLOCKED_EXTERNAL → BLOCKED_EXTERNAL
    4. DATA_MISSING → DATA_MISSING
    5. NOT_READY → NOT_READY
    6. no executed checks → NOT_READY
    7. else → PASS
    """
    if not checks:
        return NOT_READY

    has_script_error = any(c["status"] == SCRIPT_ERROR for c in checks)
    if has_script_error:
        return SCRIPT_ERROR

    if any(c["status"] == FAIL for c in checks):
        return FAIL_INTERNAL

    has_blocked = any(c["status"] == BLOCKED_EXTERNAL for c in checks)
    if has_blocked:
        return BLOCKED_EXTERNAL

    has_data_missing = any(c["status"] == DATA_MISSING for c in checks)
    if has_data_missing:
        return DATA_MISSING

    has_not_ready = any(c["status"] == NOT_READY for c in checks)
    if has_not_ready:
        return NOT_READY

    if all(c["status"] == SKIPPED for c in checks):
        return NOT_READY

    return PASS


def derive_exit_code(report: dict[str, Any]) -> int:
    """Derive exit code from report status and errors list.

    0 = PASS
    1 = CONTROLLED_NON_PASS (FAIL_INTERNAL, BLOCKED_EXTERNAL, DATA_MISSING, NOT_READY)
    2 = SCRIPT_ERROR (including errors list non-empty)
    """
    if report.get("errors") or report.get("status") == SCRIPT_ERROR:
        return 2
    if report.get("status") != PASS:
        return 1
    return 0


def make_report(
    validator: str,
    run_dir: str,
    phase: str = "",
    checks: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
    status: str | None = None,
    blocked_reason: str | None = None,
    engineering_complete: bool = False,
    promotion_eligible: bool = False,
    blockers: list[dict[str, Any] | str] | None = None,
    input_hashes: dict[str, str] | None = None,
    supersedes: list[str] | None = None,
) -> dict[str, Any]:
    """Create a complete validation report dict.

    If status is not provided, it is derived from checks.
    Exit code should be set by the caller after modifyng the report.
    """
    if not validator:
        raise ValueError("validator name is required")

    checks = checks or []
    errors = errors or []
    derived = SCRIPT_ERROR if errors else derive_status(checks)
    if status is not None and status != derived:
        raise ValueError(
            f"explicit status {status!r} conflicts with derived status {derived!r}"
        )
    final_status = derived
    if promotion_eligible and (final_status != PASS or not engineering_complete):
        raise ValueError(
            "promotion_eligible requires PASS validator status and engineering_complete=true"
        )
    passed = sum(1 for c in checks if c["status"] == PASS)
    failed = sum(1 for c in checks if c["status"] == FAIL)
    blocked = sum(1 for c in checks if c["status"] == BLOCKED_EXTERNAL)
    skipped = sum(1 for c in checks if c["status"] == SKIPPED)

    return {
        "schema_version": "sa_validation_report.v1",
        "validator": validator,
        "phase": phase,
        "run_dir": run_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "validator_exit_status": final_status,
        "engineering_complete": engineering_complete,
        "promotion_eligible": promotion_eligible,
        "blockers": blockers or [],
        "input_hashes": input_hashes or {},
        "supersedes": supersedes or [],
        "exit_code": 0,  # caller must set after all modifications
        "checks": checks,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": failed,
            "blocked_external": blocked,
            "skipped": skipped,
        },
        "errors": errors,
        "blocked_reason": blocked_reason,
    }
