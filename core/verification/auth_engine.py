"""
core/auth_engine.py
===================
Feature 9 — Authentication Decision Engine
==========================================
Combines liveness and identity scores into a single GRANT / DENY decision.

Decision flow
-------------
  1. Liveness gate  — must pass texture & motion heuristics AND neural score ≥ liveness_threshold.
  2. Identity gate  — ArcFace cosine similarity ≥ identity_threshold.
  3. Combined score — weighted fusion of both.
  4. Audit log      — every attempt is persisted, regardless of outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from db.audit_log import AuditLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass — returned to callers
# ---------------------------------------------------------------------------

@dataclass
class AuthResult:
    granted: bool
    username_claimed: str
    liveness_score: float
    identity_score: float
    combined_score: float
    decision: str           # "GRANTED" | "DENIED"
    reason: str             # human-readable explanation
    audit_id: int = -1      # DB row ID written by AuditLogger


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class AuthDecisionEngine:
    """
    Fuses liveness and identity signals and makes a final authentication decision.
    """

    def __init__(
        self,
        liveness_threshold: float = 0.50,
        identity_threshold: float = 0.45,
        liveness_weight: float = 0.40,
        identity_weight: float = 0.60,
        db_path: str = "embeddings.db",
    ) -> None:
        if abs(liveness_weight + identity_weight - 1.0) > 1e-6:
            raise ValueError("liveness_weight + identity_weight must equal 1.0")

        self.liveness_threshold = liveness_threshold
        self.identity_threshold = identity_threshold
        self.liveness_weight = liveness_weight
        self.identity_weight = identity_weight
        self.audit = AuditLogger(db_path=db_path)

        logger.info(
            "AuthDecisionEngine ready | live_thr=%.2f | id_thr=%.2f | weights=[%.2f, %.2f]",
            liveness_threshold, identity_threshold, liveness_weight, identity_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        username_claimed: str,
        liveness_score: float,
        identity_score: float,
        liveness_status: str = "",
    ) -> AuthResult:
        """
        Make an authentication decision.

        Parameters
        ----------
        username_claimed : The username the subject claims to be.
        liveness_score   : Scalar in [0, 1] from LivenessEvaluator.
        identity_score   : Cosine similarity in [−1, 1] from ArcFaceVerifier.
        liveness_status  : Optional descriptive status string from LivenessEvaluator.

        Returns
        -------
        AuthResult
        """
        # ── Gate 1 — Liveness ──────────────────────────────────────────────
        if liveness_score < self.liveness_threshold:
            reason = (
                f"Liveness check failed: score {liveness_score:.3f} < "
                f"threshold {self.liveness_threshold:.2f}"
            )
            if liveness_status:
                reason += f" ({liveness_status})"
            return self._deny(username_claimed, liveness_score, identity_score, reason)

        # ── Gate 2 — Identity ──────────────────────────────────────────────
        if identity_score < self.identity_threshold:
            reason = (
                f"Identity check failed: similarity {identity_score:.3f} < "
                f"threshold {self.identity_threshold:.2f}"
            )
            return self._deny(username_claimed, liveness_score, identity_score, reason)

        # ── Combined weighted score ────────────────────────────────────────
        combined = (
            self.liveness_weight * liveness_score
            + self.identity_weight * identity_score
        )

        reason = (
            f"Access granted to '{username_claimed}' | "
            f"liveness={liveness_score:.3f} identity={identity_score:.3f} "
            f"combined={combined:.3f}"
        )
        logger.info(reason)

        audit_id = self.audit.log(
            username_claimed=username_claimed,
            decision="GRANTED",
            deny_reason="",
            liveness_score=liveness_score,
            identity_score=identity_score,
        )

        return AuthResult(
            granted=True,
            username_claimed=username_claimed,
            liveness_score=liveness_score,
            identity_score=identity_score,
            combined_score=combined,
            decision="GRANTED",
            reason=reason,
            audit_id=audit_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deny(
        self,
        username_claimed: str,
        liveness_score: float,
        identity_score: float,
        reason: str,
    ) -> AuthResult:
        combined = (
            self.liveness_weight * liveness_score
            + self.identity_weight * identity_score
        )
        logger.warning("Auth DENIED — %s", reason)

        audit_id = self.audit.log(
            username_claimed=username_claimed,
            decision="DENIED",
            deny_reason=reason,
            liveness_score=liveness_score,
            identity_score=identity_score,
        )

        return AuthResult(
            granted=False,
            username_claimed=username_claimed,
            liveness_score=liveness_score,
            identity_score=identity_score,
            combined_score=combined,
            decision="DENIED",
            reason=reason,
            audit_id=audit_id,
        )
