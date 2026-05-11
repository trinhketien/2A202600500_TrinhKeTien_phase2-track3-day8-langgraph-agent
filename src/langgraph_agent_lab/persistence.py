"""Checkpointer adapter.

Supports three backends:
- memory: MemorySaver (default, no infrastructure needed)
- sqlite: SqliteSaver with WAL mode for concurrent reads
- postgres: PostgresSaver for production deployments
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supports memory (default), sqlite, and postgres backends.
    SQLite uses WAL mode for better concurrent read performance.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Using MemorySaver checkpointer")
        return MemorySaver()

    if kind == "sqlite":
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = database_url or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        logger.info("Using SqliteSaver checkpointer at %s (WAL mode)", db_path)
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc

        logger.info("Using PostgresSaver checkpointer")
        return PostgresSaver.from_conn_string(database_url or "")

    raise ValueError(f"Unknown checkpointer kind: {kind}")
