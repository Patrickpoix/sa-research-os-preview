# 中文阅读入口：这是 验证 的回归测试，验证“报告 / contract”相关契约；fixture 与断言只用于证明行为，不是生产逻辑的第二 owner。 主要入口：`TestReportSchema`、`TestCheckContract`、`TestStatusDerivation`、`TestExitCodeDerivation`。
"""Tests for validation report contract: schema, status derivation, exit code."""
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alpha_research_os.validation.reporting import (
    make_check, make_report, derive_status, derive_exit_code,
    PASS, FAIL, BLOCKED_EXTERNAL, SKIPPED, SCRIPT_ERROR,
    DATA_MISSING, NOT_READY,
)


class TestReportSchema:
    """Report must conform to sa_validation_report.v1 schema."""

    def test_minimal_report_has_required_fields(self):
        r = make_report(validator="test", run_dir="/tmp")
        for field in [
            "schema_version",
            "validator",
            "generated_at",
            "status",
            "validator_exit_status",
            "engineering_complete",
            "promotion_eligible",
            "blockers",
            "input_hashes",
            "supersedes",
            "exit_code",
            "checks",
            "summary",
        ]:
            assert field in r, f"missing {field}"

    def test_report_is_not_promotion_eligible_by_default(self):
        r = make_report(
            validator="test",
            run_dir="/tmp",
            checks=[make_check("executed", PASS)],
        )

        assert r["status"] == PASS
        assert r["validator_exit_status"] == PASS
        assert r["engineering_complete"] is False
        assert r["promotion_eligible"] is False

    def test_schema_version_is_correct(self):
        r = make_report(validator="test", run_dir="/tmp")
        assert r["schema_version"] == "sa_validation_report.v1"

    def test_validator_name_is_required(self):
        try:
            make_report(validator="", run_dir="/tmp")
            assert False, "should have raised"
        except ValueError:
            pass


class TestCheckContract:
    """Each check must have name, status, severity, detail."""

    def test_minimal_check_has_required_fields(self):
        c = make_check(name="test_check", status=PASS)
        for field in ["name", "status", "severity", "detail"]:
            assert field in c, f"missing {field}"

    def test_default_severity_is_info(self):
        c = make_check(name="test", status=PASS)
        assert c["severity"] == "INFO"

    def test_p0_severity_stored(self):
        c = make_check(name="test", status=FAIL, severity="P0")
        assert c["severity"] == "P0"

    def test_invalid_status_raises(self):
        try:
            make_check(name="test", status="INVALID_STATUS")
            assert False, "should have raised"
        except ValueError:
            pass

    def test_detail_accepts_string(self):
        c = make_check(name="test", status=PASS, detail="all good")
        assert c["detail"] == "all good"

    def test_evidence_path_stored(self):
        c = make_check(name="test", status=PASS, evidence={"path": "/tmp/test"})
        assert c["evidence"]["path"] == "/tmp/test"


class TestStatusDerivation:
    """Derive report status from checks."""

    def test_all_pass_means_pass(self):
        checks = [make_check("a", PASS), make_check("b", PASS)]
        assert derive_status(checks) == "PASS"

    def test_script_error_means_script_error(self):
        checks = [make_check("a", PASS), make_check("b", SCRIPT_ERROR, severity="P0")]
        assert derive_status(checks) == "SCRIPT_ERROR"

    def test_p0_fail_means_fail_internal(self):
        checks = [make_check("a", PASS), make_check("b", FAIL, severity="P0")]
        assert derive_status(checks) == "FAIL_INTERNAL"

    def test_info_fail_still_means_fail_internal(self):
        checks = [make_check("a", PASS), make_check("b", FAIL)]
        assert derive_status(checks) == "FAIL_INTERNAL"

    def test_blocked_external_propagates(self):
        checks = [make_check("a", PASS), make_check("b", BLOCKED_EXTERNAL)]
        assert derive_status(checks) == "BLOCKED_EXTERNAL"

    def test_data_missing_propagates(self):
        checks = [make_check("a", PASS), make_check("b", DATA_MISSING)]
        assert derive_status(checks) == "DATA_MISSING"

    def test_not_ready_propagates(self):
        checks = [make_check("a", PASS), make_check("b", NOT_READY)]
        assert derive_status(checks) == "NOT_READY"

    def test_skipped_does_not_change_pass(self):
        checks = [make_check("a", PASS), make_check("b", SKIPPED)]
        assert derive_status(checks) == "PASS"

    def test_priority_script_error_over_fail(self):
        checks = [make_check("a", FAIL, severity="P0"), make_check("b", SCRIPT_ERROR, severity="P0")]
        assert derive_status(checks) == "SCRIPT_ERROR"

    def test_empty_checks_list_is_not_ready(self):
        assert derive_status([]) == NOT_READY

    def test_only_skipped_checks_are_not_ready(self):
        checks = [make_check("optional", SKIPPED)]
        assert derive_status(checks) == NOT_READY


class TestExitCodeDerivation:
    """Exit code derived from checks and errors list."""

    def test_all_pass_exit_0(self):
        r = make_report("test", "/tmp", checks=[make_check("a", PASS)])
        assert derive_exit_code(r) == 0

    def test_script_error_exit_2(self):
        checks = [make_check("a", SCRIPT_ERROR, severity="P0")]
        r = make_report("test", "/tmp", checks=checks)
        assert derive_exit_code(r) == 2

    def test_fail_internal_exit_1(self):
        checks = [make_check("a", FAIL, severity="P0")]
        r = make_report("test", "/tmp", checks=checks)
        assert derive_exit_code(r) == 1

    def test_blocked_external_exit_1(self):
        checks = [make_check("a", BLOCKED_EXTERNAL)]
        r = make_report("test", "/tmp", checks=checks)
        assert derive_exit_code(r) == 1

    def test_errors_list_makes_exit_2(self):
        r = make_report("test", "/tmp", checks=[], errors=["unexpected exception"])
        assert derive_exit_code(r) == 2

    def test_data_missing_exit_1(self):
        checks = [make_check("a", DATA_MISSING)]
        r = make_report("test", "/tmp", checks=checks)
        assert derive_exit_code(r) == 1

    def test_not_ready_exit_1(self):
        checks = [make_check("a", NOT_READY)]
        r = make_report("test", "/tmp", checks=checks)
        assert derive_exit_code(r) == 1


class TestReportIntegration:
    """End-to-end report generation and JSON serialization."""

    def test_report_to_json_roundtrip(self):
        checks = [make_check("check_1", PASS, severity="P0", detail="works")]
        r = make_report("test_val", "/tmp/data", checks=checks, errors=[])
        r["exit_code"] = derive_exit_code(r)
        js = json.dumps(r)
        loaded = json.loads(js)
        assert loaded["validator"] == "test_val"
        assert loaded["status"] == "PASS"
        assert loaded["exit_code"] == 0
        assert loaded["checks"][0]["name"] == "check_1"

    def test_report_with_failure_and_errors(self):
        checks = [make_check("a", FAIL, severity="P0", detail="broken")]
        r = make_report("test", "/tmp", checks=checks, errors=["err1"])
        r["exit_code"] = derive_exit_code(r)
        # errors list makes exit 2 (SCRIPT_ERROR), overriding FAIL_INTERNAL
        assert r["status"] == SCRIPT_ERROR
        assert r["exit_code"] == 2

    def test_explicit_pass_cannot_override_failed_checks(self):
        checks = [make_check("failed", FAIL)]

        try:
            make_report("test", "/tmp", checks=checks, status=PASS)
            assert False, "explicit PASS must not override failed checks"
        except ValueError:
            pass

    def test_errors_cannot_be_promotion_eligible(self):
        checks = [make_check("executed", PASS)]

        try:
            make_report(
                "test",
                "/tmp",
                checks=checks,
                errors=["unexpected exception"],
                engineering_complete=True,
                promotion_eligible=True,
            )
            assert False, "reports with errors must not be promotion eligible"
        except ValueError:
            pass
