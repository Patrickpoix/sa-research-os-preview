# 中文阅读入口：这是 验证 的核心实现，负责“输入输出”相关职责。 主要入口：`check_path_exists`、`safe_read_json`、`safe_read_parquet_schema`。
"""Safe I/O for validators: read JSON with error handling, path checks."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def check_path_exists(path: Path) -> dict[str, Any]:
    """Check if a path exists. Returns check-like dict."""
    if path.exists():
        return {"status": "PASS", "detail": f"{path} exists"}
    return {"status": "DATA_MISSING", "detail": f"{path} not found"}


def safe_read_json(path: Path, max_size_mb: int = 500) -> dict[str, Any]:
    """Read and parse a JSON file safely.

    Returns a dict with keys: status, detail, data (if success).
    Never raises an exception (catches all and returns FAIL status).
    """
    result = {"status": "PASS", "detail": "", "data": None}

    if not path.exists():
        result["status"] = "DATA_MISSING"
        result["detail"] = f"File not found: {path}"
        return result

    size_mb = path.stat().st_size / 1e6
    if size_mb > max_size_mb:
        result["status"] = "FAIL"
        result["detail"] = f"File too large: {size_mb:.1f}MB > {max_size_mb}MB limit"
        return result

    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            result["status"] = "FAIL"
            result["detail"] = f"Empty file: {path}"
            return result
        result["data"] = __import__("json").loads(text)
        result["detail"] = f"Parsed {path.name} ({size_mb:.1f}MB)"
    except ValueError as e:
        result["status"] = "FAIL"
        result["detail"] = f"Parse error: {e}"
    except Exception as e:
        result["status"] = "FAIL"
        result["detail"] = f"Read error: {e}"

    return result


def safe_read_parquet_schema(path: Path, sample_cols: list[str] | None = None) -> dict[str, Any]:
    """Read parquet file schema safely using pyarrow.

    Returns dict with status, detail, columns (list), row_count (int).
    Does NOT use pandas.read_parquet(nrows=) which is unsupported.
    """
    result = {"status": "PASS", "detail": "", "columns": [], "row_count": 0}

    if not path.exists():
        result["status"] = "DATA_MISSING"
        result["detail"] = f"File not found: {path}"
        return result

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        schema = pf.schema
        result["columns"] = [f.name for f in schema]
        result["row_count"] = pf.metadata.num_rows
        result["detail"] = f"Schema: {len(result['columns'])} cols, {result['row_count']} rows"

        if sample_cols:
            missing = [c for c in sample_cols if c not in result["columns"]]
            if missing:
                result["status"] = "FAIL"
                result["detail"] += f" | Missing columns: {missing}"

    except ImportError:
        result["status"] = "FAIL"
        result["detail"] = "pyarrow not installed"
    except Exception as e:
        result["status"] = "FAIL"
        result["detail"] = f"Parquet read error: {e}"

    return result
