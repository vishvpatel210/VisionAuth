"""
db/audit_log.py
===============
Persistent audit trail for every authentication attempt.
Stores user, timestamp, liveness score, identity similarity, and decision.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AuditRecord:
    id: int
    username_claimed: str
    decision: str           # "GRANTED" | "DENIED"
    deny_reason: str
    liveness_score: float
    identity_score: float
    timestamp: str


class AuditLogger:
    def __init__(self, db_path: str = "embeddings.db") -> None:
        self.db_path = db_path
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_table(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                username_claimed TEXT    NOT NULL,
                decision         TEXT    NOT NULL,
                deny_reason      TEXT    DEFAULT '',
                liveness_score   REAL    NOT NULL,
                identity_score   REAL    NOT NULL,
                timestamp        TEXT    NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def log(
        self,
        username_claimed: str,
        decision: str,
        deny_reason: str,
        liveness_score: float,
        identity_score: float,
    ) -> int:
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO audit_log
               (username_claimed, decision, deny_reason, liveness_score, identity_score, timestamp)
               VALUES (?, ?, ?, ?, ?, ?);""",
            (username_claimed, decision, deny_reason,
             liveness_score, identity_score, ts),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def recent(self, limit: int = 20) -> List[AuditRecord]:
        conn = self._connect()
        rows = conn.execute(
            """SELECT id, username_claimed, decision, deny_reason,
                      liveness_score, identity_score, timestamp
               FROM audit_log ORDER BY id DESC LIMIT ?;""",
            (limit,),
        ).fetchall()
        conn.close()
        return [AuditRecord(*r) for r in rows]

    def clear(self) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM audit_log;")
        conn.commit()
        conn.close()
