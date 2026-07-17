"""Champion/challenger model registry for promotion decisions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.db import get_connection, init_schema, transaction
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

PROMOTION_THRESHOLD_KEY = "promotion_threshold"


class ModelRegistry:
    """Manage champion/challenger roles, promotion decisions, and audit history."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self.config = config or get_config()
        paths = ProjectPaths(self.config)
        self.registry_path = Path(registry_path or paths.registry_db)

        cc_cfg = self.config["champion_challenger"]
        self.primary_metric: str = cc_cfg["primary_metric"]
        self.promotion_threshold: float = float(cc_cfg["promotion_threshold"])
        self.min_recall_floor: float = float(cc_cfg["min_recall_floor"])

        self.champion: dict[str, Any] | None = None
        self.challenger: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    @classmethod
    def load(
        cls,
        config: dict[str, Any] | None = None,
        registry_path: str | Path | None = None,
    ) -> "ModelRegistry":
        """Load registry state from disk, or return a fresh instance if missing."""
        instance = cls(config=config, registry_path=registry_path)
        if instance.registry_path.exists():
            instance._load_from_db()
            logger.info("Loaded model registry from %s", instance.registry_path)
        return instance

    def _load_from_db(self) -> None:
        conn = get_connection(self.registry_path)
        try:
            init_schema(conn)
            self.champion = self._read_role(conn, "champion")
            self.challenger = self._read_role(conn, "challenger")
            threshold_row = conn.execute(
                "SELECT value FROM registry_meta WHERE key = ?",
                (PROMOTION_THRESHOLD_KEY,),
            ).fetchone()
            if threshold_row is not None:
                self.promotion_threshold = float(threshold_row["value"])
            self.history = self._read_history(conn)
        finally:
            conn.close()

    def _read_role(self, conn: sqlite3.Connection, role: str) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT model_name, metrics_json, trained_at, promoted_at FROM model_roles WHERE role = ?",
            (role,),
        ).fetchone()
        if row is None or row["model_name"] is None:
            return None
        return {
            "model_name": row["model_name"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "trained_at": row["trained_at"],
            "promoted_at": row["promoted_at"],
        }

    def _read_history(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT timestamp, action, previous_champion, new_champion,
                   champion_metric, challenger_metric, delta
            FROM promotion_history
            ORDER BY id
            """
        ).fetchall()
        return [
            {
                "timestamp": row["timestamp"],
                "action": row["action"],
                "previous_champion": row["previous_champion"],
                "new_champion": row["new_champion"],
                "champion_metric": row["champion_metric"],
                "challenger_metric": row["challenger_metric"],
                "delta": row["delta"],
            }
            for row in rows
        ]

    def save(self) -> None:
        """Persist registry state atomically."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        conn = get_connection(self.registry_path)
        try:
            init_schema(conn)
            with transaction(conn):
                self._write_roles(conn)
                self._write_meta(conn)
                self._write_history(conn, self.history)
        finally:
            conn.close()
        logger.info("Saved model registry to %s", self.registry_path)

    def _write_roles(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM model_roles")
        for role_name, role_data in (
            ("champion", self.champion),
            ("challenger", self.challenger),
        ):
            if role_data is None:
                continue
            conn.execute(
                """
                INSERT INTO model_roles (role, model_name, metrics_json, trained_at, promoted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    role_name,
                    role_data["model_name"],
                    json.dumps(role_data.get("metrics", {})),
                    role_data.get("trained_at"),
                    role_data.get("promoted_at"),
                ),
            )

    def _write_meta(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO registry_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (PROMOTION_THRESHOLD_KEY, str(self.promotion_threshold)),
        )

    def _write_history(
        self,
        conn: sqlite3.Connection,
        history: list[dict[str, Any]],
    ) -> None:
        conn.execute("DELETE FROM promotion_history")
        for entry in history:
            conn.execute(
                """
                INSERT INTO promotion_history (
                    timestamp, action, previous_champion, new_champion,
                    champion_metric, challenger_metric, delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["timestamp"],
                    entry["action"],
                    entry.get("previous_champion"),
                    entry["new_champion"],
                    entry["champion_metric"],
                    entry["challenger_metric"],
                    entry["delta"],
                ),
            )

    def get_current_champion_name(self) -> str | None:
        """Return the current champion model name, if set."""
        if self.champion is None:
            return None
        return self.champion.get("model_name")

    def get_current_challenger_name(self) -> str | None:
        """Return the current challenger model name, if set."""
        if self.challenger is None:
            return None
        return self.challenger.get("model_name")

    def _role_entry(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build a champion/challenger role record from a training result."""
        return {
            "model_name": result["model_name"],
            "metrics": result["metrics"],
            "trained_at": result.get("trained_at", _utc_now_iso()),
            "promoted_at": result.get("promoted_at"),
        }

    def evaluate_and_decide(
        self,
        champion_result: dict[str, Any],
        challenger_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare fresh results and decide whether to promote the challenger."""
        timestamp = _utc_now_iso()

        if self.champion is None:
            first, second = self._resolve_initial_roles(champion_result, challenger_result)
            self.champion = self._role_entry(first)
            self.champion["promoted_at"] = timestamp
            self.challenger = self._role_entry(second)

            champion_metric = self._metric_value(first)
            challenger_metric = self._metric_value(second)
            delta = challenger_metric - champion_metric
            history_entry = {
                "timestamp": timestamp,
                "action": "initial",
                "previous_champion": None,
                "new_champion": self.champion["model_name"],
                "champion_metric": champion_metric,
                "challenger_metric": challenger_metric,
                "delta": delta,
            }
            self.history.append(history_entry)
            self.save()
            return {
                "action": "initial",
                "champion": self.champion["model_name"],
                "challenger": self.challenger["model_name"],
                "delta": delta,
                "history_entry": history_entry,
            }

        previous_champion_entry = self.champion
        previous_champion = previous_champion_entry["model_name"]
        previous_promoted_at = previous_champion_entry.get("promoted_at")

        self.champion = self._role_entry(champion_result)
        self.challenger = self._role_entry(challenger_result)

        champion_metric = self._metric_value(champion_result)
        challenger_metric = self._metric_value(challenger_result)
        delta = challenger_metric - champion_metric

        recall = float(challenger_result["metrics"].get("recall", 0.0))
        should_promote = (
            delta >= self.promotion_threshold
            and recall >= self.min_recall_floor
        )

        if should_promote:
            self.champion, self.challenger = self.challenger, self.champion
            self.champion["promoted_at"] = timestamp
            action = "promote"
            new_champion = self.champion["model_name"]
        else:
            self.champion["promoted_at"] = previous_promoted_at or timestamp
            action = "retain"
            new_champion = previous_champion

        history_entry = {
            "timestamp": timestamp,
            "action": action,
            "previous_champion": previous_champion,
            "new_champion": new_champion,
            "champion_metric": champion_metric,
            "challenger_metric": challenger_metric,
            "delta": delta,
        }
        self.history.append(history_entry)
        self.save()

        return {
            "action": action,
            "champion": self.champion["model_name"],
            "challenger": self.challenger["model_name"],
            "delta": delta,
            "history_entry": history_entry,
        }

    def rollback(self) -> dict[str, Any]:
        """Manually swap champion and challenger without retraining."""
        if self.champion is None or self.challenger is None:
            raise ValueError("Cannot rollback before champion and challenger are initialized")

        timestamp = _utc_now_iso()
        previous_champion = self.champion["model_name"]
        self.champion, self.challenger = self.challenger, self.champion
        self.champion["promoted_at"] = timestamp

        champion_metric = self._metric_value({"metrics": self.champion["metrics"]})
        challenger_metric = self._metric_value({"metrics": self.challenger["metrics"]})
        delta = challenger_metric - champion_metric

        history_entry = {
            "timestamp": timestamp,
            "action": "manual_rollback",
            "previous_champion": previous_champion,
            "new_champion": self.champion["model_name"],
            "champion_metric": champion_metric,
            "challenger_metric": challenger_metric,
            "delta": delta,
        }
        self.history.append(history_entry)
        self.save()

        return {
            "action": "manual_rollback",
            "champion": self.champion["model_name"],
            "challenger": self.challenger["model_name"],
            "delta": delta,
            "history_entry": history_entry,
        }

    def _resolve_initial_roles(
        self,
        first_result: dict[str, Any],
        second_result: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pick champion/challenger for the first training run."""
        first_metric = self._metric_value(first_result)
        second_metric = self._metric_value(second_result)
        if first_metric >= second_metric:
            return first_result, second_result
        return second_result, first_result

    def _metric_value(self, result: dict[str, Any]) -> float:
        """Read the configured primary metric from a result dict."""
        return float(result["metrics"][self.primary_metric])


def seed_from_json_payload(registry: ModelRegistry, data: dict[str, Any]) -> None:
    """Populate an in-memory registry from a legacy JSON payload."""
    registry.champion = data.get("champion")
    registry.challenger = data.get("challenger")
    registry.promotion_threshold = float(
        data.get("promotion_threshold", registry.promotion_threshold)
    )
    registry.history = data.get("history", [])


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()
