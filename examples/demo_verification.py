"""
run_feature8.py
===============
Feature 8 — Face Verification (Enrollment & Identification) Demo
================================================================
A real-time face verification system demonstrating:
  1. Live 1:N Identification: Matches faces in the webcam against enrolled users.
  2. Live Enrollment: Press 'E' to register your face. Type a username in the terminal.

Controls
--------
  E       : Enroll a new user face
  C       : Clear database (wipe all users)
  Q / ESC : Quit
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.verification.verifier import ArcFaceVerifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 8: Verification Demo")
    p.add_argument("--source", default=0, help="Camera index or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.45, help="Cosine similarity threshold")
    p.add_argument("--log", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log)
    logger = logging.getLogger(__name__)

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # ── Initialize Verifier ───────────────────────────────────────────
    verifier = ArcFaceVerifier(db_path="embeddings.db", verification_threshold=args.threshold)
    logger.info("Enrolling model into verifier...")
    # Trigger lazy loading/warmup
    verifier._load()

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    WINDOW = "VisionAuth — Feature 8: Face Verification  [E: Enroll | C: Clear | Q: Exit]"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    logger.info("Demo windows started. Focus window and press E to enroll.")

    while True:
        packet = cap.read()
        if packet is None:
            time.sleep(0.005)
            continue

        display = packet.frame.copy()

        # ── 1:N Identification ────────────────────────────────────────
        # Identify who is in front of the camera
        username, similarity = verifier.identify_user(packet.frame)

        # Retrieve all detections from the verifier's face analysis app
        # to draw the bounding box accurately
        faces = verifier._app.get(packet.frame)
        
        if faces:
            # Highlight primary face (largest area)
            primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = map(int, primary_face.bbox)

            if username is not None:
                # Match found (Green box)
                box_color = (0, 220, 80)
                label = f"{username} ({similarity:.2f})"
            else:
                # No match / Unknown face (Red box)
                box_color = (0, 0, 220)
                label = f"UNKNOWN ({similarity:.2f})" if similarity > 0 else "UNKNOWN"

            # Draw box & Label
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y1 - 6, lh + 4)
            cv2.rectangle(display, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), (0, 0, 0), -1)
            cv2.rectangle(display, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), box_color, 1)
            cv2.putText(
                display, label, (x1 + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
            )

        # Render Controls Hints
        cv2.putText(
            display, "E: Enroll User   C: Clear Database   Q: Exit", (15, args.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            # Clear database
            verifier.db.clear_database()
            logger.info("Enrolled database wiped.")
        elif key == ord("e"):
            # Pause capture stream while entering user info in terminal
            cap.pause()
            print("\n" + "=" * 50)
            username_input = input("Enter username to enroll: ").strip()
            print("=" * 50)
            
            if username_input:
                logger.info("Registering face for '%s'...", username_input)
                # Capture fresh packet
                cap.resume()
                time.sleep(0.5)  # Let camera settle a bit
                enroll_packet = cap.read()
                
                if enroll_packet is not None:
                    success, msg = verifier.enroll_user(username_input, enroll_packet.frame)
                    print(f">> {msg}\n")
                else:
                    print(">> Enrollment Failed: Could not grab frame.\n")
            else:
                print(">> Enrollment Cancelled: Empty username.\n")
            
            cap.resume()

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 8 demo finished.")


if __name__ == "__main__":
    main()
