import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from api.app import app

client = TestClient(app)


def test_root_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "FaceShield" in response.text


def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "fps" in data
    assert "face_detected" in data


def test_portal_signup_validation():
    # Test signup with invalid file type or missing params
    response = client.post(
        "/api/portal/signup",
        data={"username": "testuser", "email": "test@example.com", "password": "password123"},
    )
    assert response.status_code == 422  # Validation error (missing file)


def test_portal_login_password_failure():
    response = client.post(
        "/api/portal/login/password",
        data={"email": "nonexistent@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_api_audit_log():
    # Fetch audit logs
    response = client.get("/api/audit")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
