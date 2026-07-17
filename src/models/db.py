"""SQLite persistence for model registry and shared application data."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with explicit transaction control."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create registry tables if they do not exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_roles (
            role TEXT PRIMARY KEY CHECK(role IN ('champion', 'challenger')),
            model_name TEXT,
            metrics_json TEXT,
            trained_at TEXT,
            promoted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS promotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            previous_champion TEXT,
            new_champion TEXT,
            champion_metric REAL,
            challenger_metric REAL,
            delta REAL
        );
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            key_prefix TEXT,
            client_name TEXT,
            scopes TEXT,
            created_at TEXT,
            revoked_at TEXT
        );
        """
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run work inside a BEGIN IMMEDIATE transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row to a plain dict."""
    if row is None:
        return None
    return dict(row)
