"""
tests/test_auth_engine.py
=========================
Unit tests for Feature 9 — AuditLogger and AuthDecisionEngine.

Run with:
    py -m pytest tests/test_auth_engine.py -v
"""

import os
import pytest

from db.audit_log import AuditLogger, AuditRecord
from core.auth_engine import AuthDecisionEngine, AuthResult


# ── AuditLogger ────────────────────────────────────────────────────────────

class TestAuditLogger:
    @pytest.fixture()
    def logger(self, tmp_path):
        db = tmp_path / "audit_test.db"
        al = AuditLogger(db_path=str(db))
        yield al
        if db.exists():
            os.remove(db)

    def test_log_granted(self, logger):
        row_id = logger.log("alice", "GRANTED", "", 0.82, 0.91)
        assert row_id > 0

    def test_log_denied(self, logger):
        row_id = logger.log("bob", "DENIED", "Liveness check failed", 0.30, 0.10)
        assert row_id > 0

    def test_recent_returns_records(self, logger):
        logger.log("alice", "GRANTED", "", 0.82, 0.91)
        logger.log("bob",   "DENIED",  "Low liveness", 0.30, 0.10)

        records = logger.recent(limit=10)
        assert len(records) == 2
        # Most recent first
        assert records[0].username_claimed == "bob"
        assert records[1].username_claimed == "alice"

    def test_recent_respects_limit(self, logger):
        for i in range(5):
            logger.log(f"user{i}", "GRANTED", "", 0.9, 0.9)

        records = logger.recent(limit=3)
        assert len(records) == 3

    def test_clear(self, logger):
        logger.log("alice", "GRANTED", "", 0.82, 0.91)
        logger.clear()
        assert logger.recent() == []

    def test_record_fields(self, logger):
        logger.log("carol", "DENIED", "Photo attack", 0.10, 0.05)
        rec = logger.recent(1)[0]

        assert isinstance(rec, AuditRecord)
        assert rec.username_claimed == "carol"
        assert rec.decision == "DENIED"
        assert rec.deny_reason == "Photo attack"
        assert rec.liveness_score == pytest.approx(0.10)
        assert rec.identity_score == pytest.approx(0.05)
        assert rec.timestamp.endswith("Z")


# ── AuthDecisionEngine ─────────────────────────────────────────────────────

class TestAuthDecisionEngine:
    @pytest.fixture()
    def engine(self, tmp_path):
        db = tmp_path / "auth_test.db"
        eng = AuthDecisionEngine(
            liveness_threshold=0.50,
            identity_threshold=0.45,
            liveness_weight=0.40,
            identity_weight=0.60,
            db_path=str(db),
        )
        yield eng
        if db.exists():
            os.remove(db)

    # ── happy path ──────────────────────────────────────────────────────

    def test_grant_when_both_pass(self, engine):
        result = engine.decide("alice", liveness_score=0.85, identity_score=0.90)

        assert isinstance(result, AuthResult)
        assert result.granted is True
        assert result.decision == "GRANTED"
        assert result.username_claimed == "alice"
        assert result.audit_id > 0
        # combined = 0.40*0.85 + 0.60*0.90 = 0.34 + 0.54 = 0.88
        assert result.combined_score == pytest.approx(0.88, abs=1e-4)

    # ── liveness gate ───────────────────────────────────────────────────

    def test_deny_on_low_liveness(self, engine):
        result = engine.decide("alice", liveness_score=0.30, identity_score=0.90)

        assert result.granted is False
        assert result.decision == "DENIED"
        assert "Liveness" in result.reason
        assert result.audit_id > 0

    def test_deny_at_liveness_boundary(self, engine):
        # Exactly at threshold => just below (0.4999…) → DENIED
        result = engine.decide("alice", liveness_score=0.4999, identity_score=0.99)
        assert result.granted is False

    def test_grant_at_liveness_threshold(self, engine):
        # Exactly at threshold → GRANTED
        result = engine.decide("alice", liveness_score=0.50, identity_score=0.99)
        assert result.granted is True

    # ── identity gate ───────────────────────────────────────────────────

    def test_deny_on_low_identity(self, engine):
        result = engine.decide("alice", liveness_score=0.90, identity_score=0.20)

        assert result.granted is False
        assert "Identity" in result.reason

    def test_deny_at_identity_boundary(self, engine):
        result = engine.decide("alice", liveness_score=0.90, identity_score=0.4499)
        assert result.granted is False

    def test_grant_at_identity_threshold(self, engine):
        result = engine.decide("alice", liveness_score=0.90, identity_score=0.45)
        assert result.granted is True

    # ── audit ───────────────────────────────────────────────────────────

    def test_every_decision_is_logged(self, engine):
        engine.decide("alice", liveness_score=0.85, identity_score=0.90)  # GRANTED
        engine.decide("bob",   liveness_score=0.10, identity_score=0.90)  # DENIED

        records = engine.audit.recent()
        assert len(records) == 2

    # ── validation ──────────────────────────────────────────────────────

    def test_invalid_weights_raise(self, tmp_path):
        with pytest.raises(ValueError, match="must equal 1.0"):
            AuthDecisionEngine(
                liveness_weight=0.60,
                identity_weight=0.60,
                db_path=str(tmp_path / "x.db"),
            )
