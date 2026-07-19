"""
run_server.py
=============
Entry point: starts the FaceShield AI FastAPI server.
"""

import argparse
import logging
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    print("=" * 60)
    print("  FaceShield AI — Biometric Authentication Server")
    print("  URL: http://127.0.0.1:8000")
    print("=" * 60)

    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
