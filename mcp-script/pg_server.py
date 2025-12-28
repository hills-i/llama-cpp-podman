from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return
    load_dotenv(override=False)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got: {raw!r}")


def _read_secret_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Secret file not found: {path}")
    value = p.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Secret file is empty: {path}")
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _import_fastmcp():
    """Import the MCP SDK's FastMCP."""

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "MCP SDK is not installed. Run: pip install -r requirements.txt"
        ) from e


def _import_psycopg():
    try:
        import psycopg  # type: ignore
        from psycopg.rows import dict_row  # type: ignore

        return psycopg, dict_row
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "psycopg is not installed. Install deps from requirements.txt before running."
        ) from e


_FORBIDDEN_SQL = {
    "alter",
    "analyze",
    "attach",
    "call",
    "checkpoint",
    "cluster",
    "comment",
    "copy",
    "create",
    "deallocate",
    "delete",
    "detach",
    "discard",
    "do",
    "drop",
    "execute",
    "explain",  # keep surface minimal; can be used to leak plan details
    "grant",
    "insert",
    "listen",
    "load",
    "lock",
    "merge",
    "notify",
    "prepare",
    "reassign",
    "refresh",
    "reindex",
    "release",
    "reset",
    "revoke",
    "savepoint",
    "security",
    "set",
    "show",
    "truncate",
    "unlisten",
    "update",
    "vacuum",
    "write",
    "into",  # blocks SELECT INTO
}


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments before validation to prevent obfuscation attacks.
    
    Handles:
    - Single-line comments: -- ... (to end of line)
    - Multi-line comments: /* ... */ (including nested)
    """
    result = []
    i = 0
    in_string = False
    string_char = None
    
    while i < len(sql):
        # Track string literals to avoid stripping comments inside strings
        if not in_string and sql[i] in ("'", '"'):
            in_string = True
            string_char = sql[i]
            result.append(sql[i])
            i += 1
        elif in_string:
            if sql[i] == string_char:
                # Check for escaped quote
                if i + 1 < len(sql) and sql[i + 1] == string_char:
                    result.append(sql[i:i + 2])
                    i += 2
                else:
                    in_string = False
                    string_char = None
                    result.append(sql[i])
                    i += 1
            else:
                result.append(sql[i])
                i += 1
        # Single-line comment
        elif sql[i:i + 2] == "--":
            # Skip to end of line
            while i < len(sql) and sql[i] != "\n":
                i += 1
            # Keep the newline for formatting
            if i < len(sql):
                result.append(" ")
                i += 1
        # Multi-line comment
        elif sql[i:i + 2] == "/*":
            i += 2
            depth = 1
            while i < len(sql) and depth > 0:
                if sql[i:i + 2] == "/*":
                    depth += 1
                    i += 2
                elif sql[i:i + 2] == "*/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            result.append(" ")  # Replace comment with space
        else:
            result.append(sql[i])
            i += 1
    
    return "".join(result)


def _normalize_and_validate_select_only(sql: str) -> str:
    """Validate SQL is a safe, read-only SELECT statement.
    
    Security measures:
    1. Unicode normalization to prevent homoglyph attacks
    2. ASCII-only enforcement
    3. Comment stripping to prevent obfuscation
    4. Keyword blocklist validation
    5. Statement structure validation via sqlparse
    """
    if not isinstance(sql, str):
        raise ValueError("sql must be a string")

    # Step 1: Normalize Unicode to catch homoglyph attacks (e.g., Cyrillic letters)
    # NFKC normalization converts compatibility characters to their canonical form
    normalized = unicodedata.normalize("NFKC", sql)
    
    # Step 2: Reject non-ASCII characters for strict safety
    # This prevents Unicode-based bypasses entirely
    if not all(ord(c) < 128 for c in normalized):
        raise ValueError("SQL must contain only ASCII characters")
    
    max_chars = _get_env_int("PG_MAX_SQL_CHARS", 20000)
    if len(normalized) > max_chars:
        raise ValueError(f"SQL is too long (max {max_chars} chars)")

    # Step 3: Strip comments before any validation
    # This prevents attackers from hiding keywords in comments
    stripped = _strip_sql_comments(normalized)
    candidate = stripped.strip()
    
    if not candidate:
        raise ValueError("SQL is empty")

    # Step 4: Handle semicolons - allow single trailing semicolon only
    if ";" in candidate:
        stripped_semi = candidate.rstrip()
        if stripped_semi.endswith(";") and ";" not in stripped_semi[:-1]:
            candidate = stripped_semi[:-1].rstrip()
        else:
            raise ValueError("Multiple statements are not allowed (semicolon detected)")

    lowered = candidate.lower()

    # Step 5: Quick safety check - forbid dangerous keywords anywhere
    pattern = r"\b(" + "|".join(sorted(re.escape(k) for k in _FORBIDDEN_SQL)) + r")\b"
    if re.search(pattern, lowered, flags=re.IGNORECASE):
        raise ValueError("Only read-only SELECT queries are allowed")

    # Step 6: Must start with SELECT or WITH (for CTEs)
    if not re.match(r"^(select|with)\b", lowered, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    # Step 7: Use sqlparse for structural validation if available
    try:
        import sqlparse  # type: ignore

        statements = [s for s in sqlparse.parse(candidate) if str(s).strip()]
        if len(statements) != 1:
            raise ValueError("Only a single SELECT statement is allowed")
        stmt = statements[0]
        stmt_type = getattr(stmt, "get_type", lambda: "UNKNOWN")()
        if str(stmt_type).upper() != "SELECT":
            # Some parsers return UNKNOWN for certain WITH queries; we already gate on WITH.
            if not lowered.startswith("with"):
                raise ValueError("Only SELECT queries are allowed")
    except ModuleNotFoundError:
        pass

    return candidate


def _get_pg_settings() -> dict[str, Any]:
    dsn = os.getenv("PG_DSN")
    if not dsn:
        host = os.getenv("PG_HOST")
        port = os.getenv("PG_PORT", "5432")
        database = os.getenv("PG_DATABASE")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")
        password_file = os.getenv("PG_PASSWORD_FILE")

        if password_file and not password:
            password = _read_secret_file(password_file)

        if host and database and user:
            user_q = quote_plus(user)
            if password:
                pwd_q = quote_plus(password)
                dsn = f"postgresql://{user_q}:{pwd_q}@{host}:{port}/{database}"
            else:
                dsn = f"postgresql://{user_q}@{host}:{port}/{database}"
        else:
            raise RuntimeError(
                "PostgreSQL connection is not configured. Set PG_DSN, or set PG_HOST/PG_PORT/PG_DATABASE/PG_USER "
                "(+ PG_PASSWORD or PG_PASSWORD_FILE)."
            )

    return {
        "dsn": dsn,
        "connect_timeout": _get_env_int("PG_CONNECT_TIMEOUT", 5),
        "statement_timeout_ms": _get_env_int("PG_STATEMENT_TIMEOUT_MS", 5000),
        "max_rows": _get_env_int("PG_MAX_ROWS", 200),
    }


def _connect_readonly(psycopg_module):
    settings = _get_pg_settings()

    # Apply server-side safety defaults.
    options = f"-c statement_timeout={settings['statement_timeout_ms']} -c default_transaction_read_only=on"

    return psycopg_module.connect(
        settings["dsn"],
        connect_timeout=settings["connect_timeout"],
        options=options,
    )


FastMCP = _import_fastmcp()
app = FastMCP("postgres-readonly")


def _sanitize_db_error(e: Exception) -> str:
    """Return a safe error message that doesn't leak sensitive details.
    
    Full exception details are logged server-side but not returned to clients.
    """
    error_type = type(e).__name__
    
    # Map known exception types to safe messages
    safe_messages = {
        "OperationalError": "Database connection error",
        "InterfaceError": "Database interface error",
        "ProgrammingError": "Invalid query syntax",
        "DataError": "Invalid data format",
        "IntegrityError": "Data integrity error",
        "QueryCanceled": "Query timed out or was canceled",
        "ConnectionTimeout": "Database connection timed out",
    }
    
    return safe_messages.get(error_type, "Database operation failed")


@app.tool()
def list_tables() -> dict[str, Any]:
    """List user tables (schema + name) excluding system schemas."""

    _load_env()
    psycopg, dict_row = _import_psycopg()

    sql = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
      AND table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name
    """.strip()

    try:
        with _connect_readonly(psycopg) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return {"tables": _json_safe(rows), "count": len(rows)}
    except Exception as e:
        logger.exception("list_tables failed")
        return {
            "error": "Failed to list tables",
            "detail": _sanitize_db_error(e),
        }


@app.tool()
def query_database(sql: str, params: dict[str, Any] | None = None, max_rows: int | None = None) -> dict[str, Any]:
    """Run a read-only SELECT query and return up to max_rows rows."""

    _load_env()
    psycopg, dict_row = _import_psycopg()

    try:
        normalized = _normalize_and_validate_select_only(sql)
    except ValueError as e:
        return {"error": "Invalid SQL", "detail": str(e)}

    settings = _get_pg_settings()
    effective_max_rows = max_rows if (max_rows is not None and max_rows > 0) else settings["max_rows"]
    effective_max_rows = min(effective_max_rows, settings["max_rows"])

    try:
        with _connect_readonly(psycopg) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(normalized, params)
                fetched = cur.fetchmany(effective_max_rows + 1)

        truncated = len(fetched) > effective_max_rows
        rows = fetched[:effective_max_rows]

        columns: list[str]
        if rows:
            columns = list(rows[0].keys())
        else:
            columns = []

        return {
            "sql": normalized,
            "columns": columns,
            "rows": _json_safe(rows),
            "row_count": len(rows),
            "truncated": truncated,
            "max_rows": effective_max_rows,
        }
    except Exception as e:
        logger.exception("query_database failed")
        return {
            "error": "Database query failed",
            "detail": _sanitize_db_error(e),
        }


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    _load_env()
    app.run()


if __name__ == "__main__":
    main()
