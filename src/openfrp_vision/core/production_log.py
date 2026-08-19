from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProductionLogRow:
    row_id: int
    created_at: str
    profile_id: str
    profile_name: str
    decision: str
    passed: bool
    frame_id: int | None
    source_node: str
    result_json: str
    checks_json: str


def default_log_db_path() -> Path:
    return Path.home() / ".local" / "share" / "openfrp_vision" / "production_logs.sqlite3"


def resolve_log_db_path(path_text: str | None) -> Path:
    if path_text:
        return Path(path_text).expanduser()
    return default_log_db_path()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=2.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=2000")
    return connection


def ensure_schema(path: Path) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS production_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                profile_name TEXT NOT NULL,
                decision TEXT NOT NULL,
                passed INTEGER NOT NULL,
                frame_id INTEGER,
                source_node TEXT NOT NULL,
                result_json TEXT NOT NULL,
                checks_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_logs_profile_time
            ON production_logs(profile_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_logs_decision_time
            ON production_logs(decision, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS production_log_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO production_log_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )


def insert_result(path: Path, result: dict[str, Any], params: dict[str, Any]) -> int:
    ensure_schema(path)
    checks = result.get("checks")
    if not isinstance(checks, list):
        checks = []
    passed = bool(result.get("passed", False))
    decision = str(result.get("decision") or ("PASS" if passed else "FAIL"))
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO production_logs(
                created_at,
                profile_id,
                profile_name,
                decision,
                passed,
                frame_id,
                source_node,
                result_json,
                checks_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(params.get("_profile_id", "")),
                str(params.get("_profile_name", "")),
                decision,
                1 if passed else 0,
                _optional_int(params.get("_frame_id")),
                str(params.get("_node_id", "")),
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                json.dumps(checks, ensure_ascii=False, sort_keys=True),
            ),
        )
        return int(cursor.lastrowid)


def query_rows(path: Path, profile_id: str = "", limit: int = 200) -> list[ProductionLogRow]:
    ensure_schema(path)
    limit = max(1, min(int(limit), 5000))
    query = (
        "SELECT id, created_at, profile_id, profile_name, decision, passed, frame_id, source_node, result_json, checks_json "
        "FROM production_logs "
    )
    params: tuple[Any, ...]
    if profile_id:
        query += "WHERE profile_id = ? "
        params = (profile_id, limit)
    else:
        params = (limit,)
    query += "ORDER BY id DESC LIMIT ?"
    with _connect(path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        ProductionLogRow(
            row_id=int(row[0]),
            created_at=str(row[1]),
            profile_id=str(row[2]),
            profile_name=str(row[3]),
            decision=str(row[4]),
            passed=bool(row[5]),
            frame_id=int(row[6]) if row[6] is not None else None,
            source_node=str(row[7]),
            result_json=str(row[8]),
            checks_json=str(row[9]),
        )
        for row in rows
    ]


def month_day_counts(path: Path, profile_id: str, year: int, month: int) -> dict[str, dict[str, int]]:
    ensure_schema(path)
    start = f"{int(year):04d}-{int(month):02d}-01"
    if int(month) == 12:
        end = f"{int(year) + 1:04d}-01-01"
    else:
        end = f"{int(year):04d}-{int(month) + 1:02d}-01"
    query = (
        "SELECT substr(created_at, 1, 10) AS created_date, "
        "COUNT(*) AS total, "
        "SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END) AS ok, "
        "SUM(CASE WHEN passed=0 THEN 1 ELSE 0 END) AS ng "
        "FROM production_logs "
        "WHERE substr(created_at, 1, 10) >= ? AND substr(created_at, 1, 10) < ? "
    )
    params: list[Any] = [start, end]
    if profile_id:
        query += "AND profile_id = ? "
        params.append(profile_id)
    query += "GROUP BY created_date"
    with _connect(path) as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return {
        str(row[0]): {
            "total": int(row[1] or 0),
            "ok": int(row[2] or 0),
            "ng": int(row[3] or 0),
        }
        for row in rows
    }


def query_rows_for_day(path: Path, profile_id: str, date_text: str, limit: int = 1000) -> list[ProductionLogRow]:
    ensure_schema(path)
    limit = max(1, min(int(limit), 5000))
    query = (
        "SELECT id, created_at, profile_id, profile_name, decision, passed, frame_id, source_node, result_json, checks_json "
        "FROM production_logs "
        "WHERE substr(created_at, 1, 10) = ? "
    )
    params: list[Any] = [date_text]
    if profile_id:
        query += "AND profile_id = ? "
        params.append(profile_id)
    query += "ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect(path) as connection:
        rows = connection.execute(query, tuple(params)).fetchall()
    return [
        ProductionLogRow(
            row_id=int(row[0]),
            created_at=str(row[1]),
            profile_id=str(row[2]),
            profile_name=str(row[3]),
            decision=str(row[4]),
            passed=bool(row[5]),
            frame_id=int(row[6]) if row[6] is not None else None,
            source_node=str(row[7]),
            result_json=str(row[8]),
            checks_json=str(row[9]),
        )
        for row in rows
    ]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
