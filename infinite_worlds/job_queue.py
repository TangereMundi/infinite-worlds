"""JobQueue: persistent SQLite queue. Safe to kill at any moment.

States: pending -> running -> done | failed | skipped
On start-up any job left 'running' (crash / power loss / reboot) is reset to
'pending'; the render resumes from its last completed chunk.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class JobQueue:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                path TEXT NOT NULL,
                generation_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                output_path TEXT,
                progress REAL NOT NULL DEFAULT 0,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                UNIQUE(content_hash, generation_id)
            );
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER, ts REAL, level TEXT, message TEXT
            );
        """)

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------------- recovery
    def recover(self) -> int:
        cur = self.db.execute(
            "UPDATE jobs SET status='pending', updated=? WHERE status='running'", (time.time(),))
        return cur.rowcount

    # -------------------------------------------------------------------- add
    def add(self, path: Path, content_hash: str, generation_id: str) -> tuple[int, bool]:
        """Returns (job_id, newly_added). Duplicate (hash, generation) is never re-queued."""
        now = time.time()
        row = self.db.execute("SELECT id FROM jobs WHERE content_hash=? AND generation_id=?",
                              (content_hash, generation_id)).fetchone()
        if row:
            return row["id"], False
        cur = self.db.execute(
            "INSERT INTO jobs(content_hash,path,generation_id,created,updated) VALUES(?,?,?,?,?)",
            (content_hash, str(path), generation_id, now, now))
        return cur.lastrowid, True

    # ------------------------------------------------------------------- work
    def claim_next(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
            if row:
                self.db.execute("UPDATE jobs SET status='running', attempts=attempts+1, updated=? WHERE id=?",
                                (time.time(), row["id"]))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return row

    def progress(self, job_id: int, frac: float) -> None:
        self.db.execute("UPDATE jobs SET progress=?, updated=? WHERE id=?",
                        (round(frac, 4), time.time(), job_id))

    def finish(self, job_id: int, status: str, output_path: str | None = None, error: str | None = None):
        self.db.execute("UPDATE jobs SET status=?, output_path=?, error=?, progress=?, updated=? WHERE id=?",
                        (status, output_path, error, 1.0 if status == "done" else 0, time.time(), job_id))

    def event(self, job_id: int | None, level: str, message: str) -> None:
        self.db.execute("INSERT INTO events(job_id,ts,level,message) VALUES(?,?,?,?)",
                        (job_id, time.time(), level, message))

    # ----------------------------------------------------------------- views
    def counts(self) -> dict:
        return {r["status"]: r["n"] for r in
                self.db.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}

    def list(self, status: str | None = None):
        q = "SELECT * FROM jobs" + (" WHERE status=?" if status else "") + " ORDER BY id"
        return self.db.execute(q, (status,) if status else ()).fetchall()

    def retry_failed(self) -> int:
        return self.db.execute("UPDATE jobs SET status='pending', error=NULL WHERE status='failed'").rowcount
