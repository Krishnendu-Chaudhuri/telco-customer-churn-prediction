"""Per-client API key management backed by registry.db."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from src.models.db import get_connection, init_schema, transaction
from src.utils.api_settings import get_api_settings
from src.utils.config import get_config
from src.utils.paths import ProjectPaths

KEY_PREFIX_LENGTH = 8
SCOPE_PREDICT = "predict"
SCOPE_ADMIN = "admin"
ALLOWED_SCOPES = frozenset({SCOPE_PREDICT, SCOPE_ADMIN})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_key(raw_key: str) -> str:
    """Return the sha256 hex digest of an API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _db_path():
    return ProjectPaths(get_config()).registry_db


def normalize_scopes(scopes: str) -> str:
    """Validate and canonicalize a comma-separated scope string."""
    parts = {part.strip() for part in scopes.split(",") if part.strip()}
    unknown = parts - ALLOWED_SCOPES
    if unknown:
        raise ValueError(f"Unsupported scopes: {', '.join(sorted(unknown))}")
    if not parts:
        raise ValueError("At least one scope is required")
    ordered = [scope for scope in (SCOPE_PREDICT, SCOPE_ADMIN) if scope in parts]
    return ",".join(ordered)


def resolve_scopes(raw_key: str | None) -> set[str] | None:
    """Return scopes for a key, or None when the key is invalid or revoked."""
    if not raw_key:
        return None

    expected_key = get_api_settings().CHURN_API_KEY
    if expected_key and hmac.compare_digest(raw_key, expected_key):
        return {SCOPE_PREDICT, SCOPE_ADMIN}

    key_hash = hash_key(raw_key)
    db_path = _db_path()
    if not db_path.exists():
        return None

    conn = get_connection(db_path)
    try:
        init_schema(conn)
        row = conn.execute(
            "SELECT key_hash, scopes FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
            (key_hash,),
        ).fetchone()
        if row is None or not hmac.compare_digest(row["key_hash"], key_hash):
            return None
        return {part.strip() for part in row["scopes"].split(",") if part.strip()}
    finally:
        conn.close()


def key_has_scope(raw_key: str | None, required: str) -> bool:
    """Return True when the key is valid and includes the required scope."""
    scopes = resolve_scopes(raw_key)
    if scopes is None:
        return False
    if required == SCOPE_PREDICT:
        return SCOPE_PREDICT in scopes or SCOPE_ADMIN in scopes
    if required == SCOPE_ADMIN:
        return SCOPE_ADMIN in scopes
    return False


def issue_key(client_name: str, scopes: str = SCOPE_PREDICT) -> str:
    """Create a new API key and return the plaintext value once."""
    normalized_scopes = normalize_scopes(scopes)
    raw_key = secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    key_prefix = raw_key[:KEY_PREFIX_LENGTH]
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        init_schema(conn)
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO api_keys (key_hash, key_prefix, client_name, scopes, created_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (key_hash, key_prefix, client_name, normalized_scopes, _utc_now_iso()),
            )
    finally:
        conn.close()
    return raw_key


def verify_key(raw_key: str | None) -> bool:
    """Return True when the key matches an active client key."""
    return resolve_scopes(raw_key) is not None


def has_active_keys() -> bool:
    """Return True when at least one non-revoked client key exists."""
    db_path = _db_path()
    if not db_path.exists():
        return False
    conn = get_connection(db_path)
    try:
        init_schema(conn)
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM api_keys WHERE revoked_at IS NULL"
        ).fetchone()
        return bool(row and row["count"] > 0)
    finally:
        conn.close()


def revoke_by_prefix(prefix: str) -> int:
    """Revoke keys whose issued prefix matches; returns rows updated."""
    if not prefix:
        return 0
    db_path = _db_path()
    if not db_path.exists():
        return 0
    conn = get_connection(db_path)
    revoked = 0
    try:
        init_schema(conn)
        with transaction(conn):
            rows = conn.execute(
                """
                SELECT key_hash FROM api_keys
                WHERE revoked_at IS NULL AND key_prefix LIKE ?
                """,
                (f"{prefix}%",),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE api_keys SET revoked_at = ? WHERE key_hash = ?",
                    (_utc_now_iso(), row["key_hash"]),
                )
                revoked += 1
    finally:
        conn.close()
    return revoked


def list_keys() -> list[dict[str, str | None]]:
    """Return metadata for stored API keys without exposing hashes."""
    db_path = _db_path()
    if not db_path.exists():
        return []
    conn = get_connection(db_path)
    try:
        init_schema(conn)
        rows = conn.execute(
            """
            SELECT key_prefix, client_name, scopes, created_at, revoked_at
            FROM api_keys
            ORDER BY created_at
            """
        ).fetchall()
        return [
            {
                "key_prefix": row["key_prefix"],
                "client_name": row["client_name"],
                "scopes": row["scopes"],
                "created_at": row["created_at"],
                "revoked_at": row["revoked_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()
