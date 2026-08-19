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
    serial_text: str = ""
    serial_value: int | None = None
    serial_start: int = 0
    serial_end: int = 0


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
        _ensure_column(connection, "production_logs", "serial_text", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "production_logs", "serial_value", "INTEGER")
        _ensure_column(connection, "production_logs", "serial_start", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "production_logs", "serial_end", "INTEGER NOT NULL DEFAULT 0")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_logs_profile_time
            ON production_logs(profile_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_logs_profile_serial
            ON production_logs(profile_id, serial_text)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS serial_node_state (
                profile_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                last_text TEXT NOT NULL,
                last_value INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, node_id, kind)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS production_serials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                production_log_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                profile_name TEXT NOT NULL,
                source_node TEXT NOT NULL,
                serial_text TEXT NOT NULL,
                serial_effective TEXT NOT NULL,
                serial_value INTEGER,
                serial_start INTEGER NOT NULL,
                serial_end INTEGER NOT NULL,
                serial_length INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_serials_profile_time
            ON production_serials(profile_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_production_serials_profile_value
            ON production_serials(profile_id, serial_value)
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
    serial = _serial_from_result(result, checks)
    passed = bool(result.get("passed", False))
    decision = str(result.get("decision") or ("PASS" if passed else "FAIL"))
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect(path) as connection:
        profile_id = str(params.get("_profile_id", ""))
        profile_name = str(params.get("_profile_name", ""))
        source_node = str(params.get("_node_id", ""))
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
                checks_json,
                serial_text,
                serial_value,
                serial_start,
                serial_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                profile_id,
                profile_name,
                decision,
                1 if passed else 0,
                _optional_int(params.get("_frame_id")),
                source_node,
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                json.dumps(checks, ensure_ascii=False, sort_keys=True),
                str(serial.get("text", "")),
                _optional_int(serial.get("value")),
                int(serial.get("start", 0) or 0),
                int(serial.get("end", 0) or 0),
            ),
        )
        row_id = int(cursor.lastrowid)
        serial_text = str(serial.get("text", ""))
        if serial_text:
            connection.execute(
                """
                INSERT INTO production_serials(
                    production_log_id,
                    created_at,
                    profile_id,
                    profile_name,
                    source_node,
                    serial_text,
                    serial_effective,
                    serial_value,
                    serial_start,
                    serial_end,
                    serial_length
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    created_at,
                    profile_id,
                    profile_name,
                    str(serial.get("node_id", source_node)),
                    serial_text,
                    str(serial.get("effective", "")),
                    _optional_int(serial.get("value")),
                    int(serial.get("start", 0) or 0),
                    int(serial.get("end", 0) or 0),
                    int(serial.get("length", 0) or 0),
                ),
            )
        return row_id


def query_rows(path: Path, profile_id: str = "", limit: int = 200) -> list[ProductionLogRow]:
    ensure_schema(path)
    limit = max(1, min(int(limit), 5000))
    query = (
        "SELECT id, created_at, profile_id, profile_name, decision, passed, frame_id, source_node, result_json, checks_json, "
        "serial_text, serial_value, serial_start, serial_end "
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
            serial_text=str(row[10] or ""),
            serial_value=int(row[11]) if row[11] is not None else None,
            serial_start=int(row[12] or 0),
            serial_end=int(row[13] or 0),
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
        "SELECT id, created_at, profile_id, profile_name, decision, passed, frame_id, source_node, result_json, checks_json, "
        "serial_text, serial_value, serial_start, serial_end "
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
            serial_text=str(row[10] or ""),
            serial_value=int(row[11]) if row[11] is not None else None,
            serial_start=int(row[12] or 0),
            serial_end=int(row[13] or 0),
        )
        for row in rows
    ]


def serial_state(path: Path, profile_id: str, node_id: str, kind: str) -> dict[str, Any] | None:
    ensure_schema(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT last_text, last_value, updated_at
            FROM serial_node_state
            WHERE profile_id = ? AND node_id = ? AND kind = ?
            """,
            (profile_id, node_id, kind),
        ).fetchone()
    if row is None:
        return None
    return {"text": str(row[0]), "value": int(row[1]), "updated_at": str(row[2])}


def save_serial_state(path: Path, profile_id: str, node_id: str, kind: str, text: str, value: int) -> None:
    ensure_schema(path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO serial_node_state(profile_id, node_id, kind, last_text, last_value, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, node_id, kind)
            DO UPDATE SET last_text=excluded.last_text, last_value=excluded.last_value, updated_at=excluded.updated_at
            """,
            (profile_id, node_id, kind, text, int(value), now),
        )


def reset_serial_state(path: Path, profile_id: str, node_id: str, kind: str | None = None) -> int:
    ensure_schema(path)
    with _connect(path) as connection:
        if kind:
            cursor = connection.execute(
                "DELETE FROM serial_node_state WHERE profile_id = ? AND node_id = ? AND kind = ?",
                (profile_id, node_id, kind),
            )
        else:
            cursor = connection.execute(
                "DELETE FROM serial_node_state WHERE profile_id = ? AND node_id = ?",
                (profile_id, node_id),
            )
        return int(cursor.rowcount or 0)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _serial_from_result(result: dict[str, Any], checks: list[Any]) -> dict[str, Any]:
    serial = result.get("serial")
    if isinstance(serial, dict):
        return serial
    for check in checks:
        if isinstance(check, dict) and isinstance(check.get("serial"), dict):
            return dict(check["serial"])
    return {}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
