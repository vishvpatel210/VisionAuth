"""
api/app.py
==========
FastAPI server for FaceShield AI — CAPTCHA-Free Biometric Authentication.
Serves the web UI and provides REST endpoints for:
  - Secure Portal: signup (face enrollment) + 2-step login (password + face CAPTCHA)
  - Admin HUD: face enrollment, verification, pipeline status, audit log
  - Video: MJPEG live camera feed
"""

from __future__ import annotations

import io
import logging
import os
import sys
import hashlib
import secrets
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent   # project root
sys.path.insert(0, str(BASE_DIR))

DB_PATH    = str(BASE_DIR / "embeddings.db")
STATIC_DIR = Path(__file__).parent / "static"

# ── Core imports ──────────────────────────────────────────────────────────────
from db.db_manager import DatabaseManager
from db.audit_log  import AuditLogger
from core.verification.verifier    import ArcFaceVerifier
from core.verification.auth_engine import AuthDecisionEngine
from api.pipeline import get_global_models, PipelineSession, stop_pipeline

logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="FaceShield AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Singletons (lazy-loaded on first use) ─────────────────────────────────────
_verifier: Optional[ArcFaceVerifier]      = None
_engine:   Optional[AuthDecisionEngine]   = None
_db:       Optional[DatabaseManager]      = None
_audit:    Optional[AuditLogger]          = None


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_verifier() -> ArcFaceVerifier:
    global _verifier
    if _verifier is None:
        _verifier = ArcFaceVerifier(db_path=DB_PATH)
        _verifier._load()
    return _verifier


def get_engine() -> AuthDecisionEngine:
    global _engine
    if _engine is None:
        _engine = AuthDecisionEngine(db_path=DB_PATH)
    return _engine


def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager(db_path=DB_PATH)
    return _db


def get_audit() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger(db_path=DB_PATH)
    return _audit


def _decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Initializing global models...")
    get_global_models(db_path=DB_PATH)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Stopping pipeline worker …")
    stop_pipeline()


# ── Static / Root ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── WebSocket Stream ──────────────────────────────────────────────────────────
import base64

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected.")
    models = get_global_models(db_path=DB_PATH)
    session = PipelineSession(models)
    
    try:
        while True:
            # Receive frame as base64 or bytes from client
            text_data = await websocket.receive_text()
            # typically client sends "data:image/jpeg;base64,..."
            if "," in text_data:
                header, encoded = text_data.split(",", 1)
            else:
                encoded = text_data
            
            data = base64.b64decode(encoded)
            frame = _decode_image(data)
            
            # Process frame and return state JSON
            state_dict = session.process_frame(frame)
            await websocket.send_json(state_dict)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.close()
        except:
            pass



# ── Admin: Enroll Face ────────────────────────────────────────────────────────
@app.post("/api/enroll")
async def api_enroll(
    username: str = Form(...),
    image:    UploadFile = File(...),
):
    data = await image.read()
    img  = _decode_image(data)

    verifier = get_verifier()
    success, msg = verifier.enroll_user(username=username, frame=img)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"message": f"User '{username}' enrolled successfully."}


# ── Admin: Verify Face (one-shot) ─────────────────────────────────────────────
@app.post("/api/verify")
async def api_verify(
    username: str = Form(...),
    image:    UploadFile = File(...),
):
    data = await image.read()
    img  = _decode_image(data)

    verifier = get_verifier()
    matched, similarity, msg = verifier.verify_identity(frame=img, username=username)
    return {"similarity": round(float(similarity), 4), "matched": matched}


# ── Admin: Audit Log ──────────────────────────────────────────────────────────
@app.get("/api/audit")
async def api_audit(limit: int = 15):
    records = get_audit().recent(limit=limit)
    return [
        {
            "id":               r.id,
            "username_claimed": r.username_claimed,
            "decision":         r.decision,
            "deny_reason":      r.deny_reason,
            "liveness_score":   round(r.liveness_score, 4),
            "identity_score":   round(r.identity_score, 4),
            "timestamp":        r.timestamp,
        }
        for r in records
    ]


@app.delete("/api/audit")
async def api_clear_audit():
    get_audit().clear()
    return {"message": "Audit log cleared."}


@app.delete("/api/users")
async def api_clear_users():
    get_db().clear_database()
    return {"message": "All enrolled users cleared."}


from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# ══════════════════════════════════════════════════════════════════════════════
# Secure Portal Endpoints (FaceShield AI Login / Signup flow)
# ══════════════════════════════════════════════════════════════════════════════

# ── Portal: Register new account (step 1 + face enrollment) ───────────────────
@app.post("/api/portal/signup")
async def portal_signup(
    username: str      = Form(...),
    email:    str      = Form(...),
    password: str      = Form(""),
    image:    UploadFile = File(...),
):
    email = email.lower().strip()

    if get_db().get_portal_user(email) is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    data = await image.read()
    img  = _decode_image(data)

    # Enroll face in the biometric DB under the username
    verifier = get_verifier()
    success, msg = verifier.enroll_user(username=username, frame=img)
    if not success:
        raise HTTPException(status_code=400, detail=f"Face enrollment failed: {msg}")

    # Store credentials persistently in SQLite
    pw_hash = _hash_pw(password) if password else "*google_auth*"
    get_db().register_portal_user(email, username, pw_hash)

    logger.info("Portal signup: user=%s email=%s", username, email)
    return {"message": f"Account created successfully! Welcome, {username}."}


# ── Portal: Step 1 — Verify email + password ──────────────────────────────────
@app.post("/api/portal/login/password")
async def portal_login_password(
    email:    str = Form(...),
    password: str = Form(...),
):
    email = email.lower().strip()
    user  = get_db().get_portal_user(email)

    if user is None or user["password_hash"] != _hash_pw(password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return {"email": email, "username": user["username"], "message": "Credentials verified."}


# ── Portal: Step 1 — Verify Google Token ──────────────────────────────────────
@app.post("/api/portal/login/google")
async def portal_login_google(
    credential: str = Form(...),
):
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not google_client_id:
        raise HTTPException(status_code=500, detail="Google Client ID not configured on server.")

    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(credential, google_requests.Request(), google_client_id)
        email = idinfo.get("email", "").lower().strip()
    except Exception as e:
        logger.error("Google Auth failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid Google token.")

    user = get_db().get_portal_user(email)
    if user is None:
        # User not found -> must sign up to enroll face
        raise HTTPException(status_code=404, detail="Account not found. Please Sign Up to enroll your face first.")

    return {"email": email, "username": user["username"], "message": "Google Login verified. Proceed to face check."}


# /api/portal/login/face-stream removed because WebSocket now handles streaming natively


# ── Portal: Step 2B — Verify face from uploaded snapshot ──────────────────────
@app.post("/api/portal/login/face-upload")
async def portal_login_face_upload(
    email: str       = Form(...),
    image: UploadFile = File(...),
):
    email = email.lower().strip()
    user  = get_db().get_portal_user(email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    username = user["username"]

    data = await image.read()
    img  = _decode_image(data)

    verifier = get_verifier()
    matched_user, similarity = verifier.identify_user(img)

    if matched_user == username and similarity >= 0.45:
        logger.info("Portal face-upload login granted for %s (sim=%.3f)", username, similarity)
        return {"success": True, "username": username, "similarity": round(float(similarity), 4)}

    raise HTTPException(
        status_code=401,
        detail=f"Face does not match registered account (similarity={similarity:.3f})."
    )
