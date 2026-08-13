"""A durable count of requests made, so daily allowances can be enforced.

Free tiers with a hard daily ceiling are easy to exhaust by accident: one
benchmark run started without thinking costs a day of access. Counting in memory
would not survive the process, so the count lives in SQLite next to everything
else this project records.

Only what the budget guard needs is stored here. The richer per-attempt log used
for analysis is a separate table with its own concerns; keeping them apart means
the guard cannot be broken by a change to the analysis schema, and it works from
the very first request rather than once logging is fully built out.

Days are counted in UTC because that is when provider quotas reset. A local-time
boundary would drift against the provider and silently allow overspend.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT NOT NULL,
    model       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_provider_day
    ON requests (provider, created_at);
"""


def utc_day() -> str:
    """Return today's date in UTC as ``YYYY-MM-DD``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RequestLog:
    """Records outbound requests and answers "how many today?"."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Args:
            db_path: SQLite file. Parent directories are created as needed.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # One connection for the object's lifetime. Recording happens on the
        # request path, and reconnecting per call costs more than the insert.
        # ``sqlite3``'s context manager commits but does not close, so the
        # per-call form also leaks handles until the interpreter collects them.
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            # The benchmark may record from a worker thread; SQLite is
            # serialised by default, so sharing one connection is safe.
            check_same_thread=False,
        )
        # Write-ahead logging keeps a metrics query from blocking the run that
        # is producing the data. It is a property of the file, so it is set
        # once rather than on every connection.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Return the connection, for callers that need direct SQL access."""
        return self._conn

    def record(self, provider: str, model: str | None = None) -> None:
        """Record one outbound request.

        Called for every attempt, including failures. A request that errored
        still consumed the allowance, so excluding it would let the guard
        overspend exactly when things are going wrong.
        """
        self._conn.execute(
            "INSERT INTO requests (provider, model, created_at) VALUES (?, ?, ?)",
            (provider, model, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def count_today(self, provider: str) -> int:
        """Return how many requests have gone to a provider today, in UTC."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM requests "
            "WHERE provider = ? AND substr(created_at, 1, 10) = ?",
            (provider, utc_day()),
        ).fetchone()
        return int(row[0]) if row else 0

    def usage_today(self) -> dict[str, int]:
        """Return today's request count for every provider seen, in UTC."""
        rows = self._conn.execute(
            "SELECT provider, COUNT(*) FROM requests "
            "WHERE substr(created_at, 1, 10) = ? GROUP BY provider",
            (utc_day(),),
        ).fetchall()
        return {provider: int(count) for provider, count in rows}

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "RequestLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
