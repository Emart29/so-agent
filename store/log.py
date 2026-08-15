"""Records every attempt, so the failure rate can be measured rather than felt.

One row per *attempt*, not per request. A request that succeeded on its third
try is not the same event as one that succeeded immediately, and collapsing them
loses the number this project exists to produce: how often enforcement works
unaided.

Raw output is stored on failure. Debugging a parse failure without the bytes
that failed is guesswork, and those bytes are also the input to any later
question about *how* models fail rather than just how often — the malformed
fixtures in the test suite came from this table.

This sits alongside the request counter rather than replacing it. The counter
backs a budget guard that has to work from the first request; this backs
analysis. Keeping them separate means a change to the analysis schema cannot
break the guard.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT    NOT NULL,
    attempt_index         INTEGER NOT NULL,
    created_at            TEXT    NOT NULL,
    provider              TEXT    NOT NULL,
    model                 TEXT    NOT NULL,
    tier                  TEXT    NOT NULL,
    requested_tier        TEXT,
    downgraded_from       TEXT,
    schema_name           TEXT    NOT NULL,
    schema_difficulty     TEXT,
    success               INTEGER NOT NULL,
    failure_type          TEXT,
    failure_detail        TEXT,
    raw_output            TEXT,
    recovered_by_extraction INTEGER NOT NULL DEFAULT 0,
    repaired_from         INTEGER,
    prompt_tokens         INTEGER,
    completion_tokens     INTEGER,
    latency_ms            REAL,
    max_tokens            INTEGER,
    critic_verdict        TEXT,
    critic_reason         TEXT
);
CREATE INDEX IF NOT EXISTS idx_attempts_group
    ON attempts (provider, model, tier, failure_type);
CREATE INDEX IF NOT EXISTS idx_attempts_run
    ON attempts (run_id, attempt_index);
"""

COLUMNS = (
    "run_id", "attempt_index", "created_at", "provider", "model", "tier",
    "requested_tier", "downgraded_from", "schema_name", "schema_difficulty",
    "success", "failure_type", "failure_detail", "raw_output",
    "recovered_by_extraction", "repaired_from", "prompt_tokens",
    "completion_tokens", "latency_ms", "max_tokens", "critic_verdict",
    "critic_reason",
)


@dataclass
class AttemptRow:
    """One generation attempt, as stored."""

    run_id: str
    attempt_index: int
    provider: str
    model: str
    tier: str
    schema_name: str
    success: bool
    created_at: str = ""
    requested_tier: str | None = None
    downgraded_from: str | None = None
    schema_difficulty: str | None = None
    failure_type: str | None = None
    failure_detail: str | None = None
    raw_output: str | None = None
    recovered_by_extraction: bool = False
    repaired_from: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    max_tokens: int | None = None
    critic_verdict: str | None = None
    critic_reason: str | None = None

    def as_row(self) -> tuple:
        return (
            self.run_id,
            self.attempt_index,
            self.created_at or datetime.now(timezone.utc).isoformat(),
            self.provider,
            self.model,
            self.tier,
            self.requested_tier,
            self.downgraded_from,
            self.schema_name,
            self.schema_difficulty,
            int(self.success),
            self.failure_type,
            self.failure_detail,
            self.raw_output,
            int(self.recovered_by_extraction),
            self.repaired_from,
            self.prompt_tokens,
            self.completion_tokens,
            self.latency_ms,
            self.max_tokens,
            self.critic_verdict,
            self.critic_reason,
        )


def new_run_id() -> str:
    """Return an identifier grouping the attempts of one logical request."""
    return uuid.uuid4().hex[:16]


class AttemptLog:
    """Stores attempts and reads them back for analysis."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, timeout=10.0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, row: AttemptRow) -> None:
        """Store one attempt."""
        placeholders = ", ".join("?" * len(COLUMNS))
        self._conn.execute(
            f"INSERT INTO attempts ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            row.as_row(),
        )
        self._conn.commit()

    def record_many(self, rows: list[AttemptRow]) -> None:
        """Store several attempts in one transaction."""
        if not rows:
            return
        placeholders = ", ".join("?" * len(COLUMNS))
        self._conn.executemany(
            f"INSERT INTO attempts ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            [row.as_row() for row in rows],
        )
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Run a read query against the attempts table."""
        return self._conn.execute(sql, params).fetchall()

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])

    def runs(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT run_id FROM attempts ORDER BY id"
        ).fetchall()
        return [r["run_id"] for r in rows]

    def attempts_for(self, run_id: str) -> list[sqlite3.Row]:
        """Return one run's attempts in order."""
        return self._conn.execute(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY attempt_index",
            (run_id,),
        ).fetchall()

    def combinations(self) -> list[tuple[str, str, str]]:
        """Return every (provider, model, tier) that has data."""
        rows = self._conn.execute(
            "SELECT DISTINCT provider, model, tier FROM attempts "
            "ORDER BY provider, model, tier"
        ).fetchall()
        return [(r["provider"], r["model"], r["tier"]) for r in rows]

    def export_jsonl(self, path: str | Path) -> int:
        """Write every attempt as JSON lines.

        SQLite is right for querying; JSONL is right for shipping a result set
        alongside an article so a reader can check the numbers.
        """
        rows = self._conn.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row)) + "\n")
        return len(rows)

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        for row in self._conn.execute("SELECT * FROM attempts ORDER BY id"):
            yield dict(row)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "AttemptLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
