"""
run_feature9.py
===============
Feature 9 — Full Authentication Pipeline Demo
=============================================
Chains all 9 features:
  Capture → Detect → Track → Align → Extract Features → Fuse (Transformer)
  → Liveness Evaluation → Identity Verification → Auth Decision Engine
  → Live HUD with GRANTED / DENIED banner + scrolling audit log

Controls
--------
  E       : Enroll current face under a username (type in terminal)
  A       : View recent audit log in terminal
  C       : Clear enrolled users + audit log
  Q / ESC : Quit
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np
import torch

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.capture.tracker import ByteTracker, select_primary_face
from core.alignment.aligner import LandmarkFreeAligner
from core.features.feature_rgb import RGBFeatureExtractor
from core.features.feature_flow import MotionEncoder, OpticalFlowExtractor
from core.features.feature_texture import TextureEncoder, TextureFeatureExtractor
from core.features.fusion import TemporalMultiModalFusionTransformer
from core.liveness.liveness import LivenessHead, LivenessHeuristics, LivenessEvaluator
from core.verification.verifier import ArcFaceVerifier
from core.verification.auth_engine import AuthDecisionEngine, AuthResult


# ── Color palette ─────────────────────────────────────────────────────────
_GREEN   = (0, 210, 80)
_RED     = (30, 30, 220)
_YELLOW  = (0, 200, 255)
_WHITE   = (240, 240, 240)
_BLACK   = (0, 0, 0)
_CYAN    = (220, 200, 0)
_MAGENTA = (220, 0, 220)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Full Auth Pipeline")
    p.add_argument("--backend", choices=["retinaface", "yolo"], default="retinaface")
    p.add_argument("--source",  default=0)
    p.add_argument("--width",   type=int,   default=640)
    p.add_argument("--height",  type=int,   default=480)
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--conf",    type=float, default=0.5)
    p.add_argument("--seq-len", type=int,   default=10)
    p.add_argument("--live-thr",  type=float, default=0.50)
    p.add_argument("--id-thr",    type=float, default=0.45)
    p.add_argument("--log",     default="INFO")
    return p.parse_args()


def build_detector(args):
    if args.backend == "retinaface":
        from core.detection.backends.retinaface_backend import RetinaFaceDetector
        return RetinaFaceDetector(det_thresh=args.conf)
    from core.detection.backends.yolo_backend import YOLODetector
    return YOLODetector(conf_thresh=args.conf)


def draw_banner(frame: np.ndarray, result: AuthResult, h: int, w: int) -> None:
    """Draw the GRANTED / DENIED decision banner at the bottom of the frame."""
    color = _GREEN if result.granted else _RED
    label = f"{'ACCESS GRANTED' if result.granted else 'ACCESS DENIED'} — {result.username_claimed}"
    cv2.rectangle(frame, (0, h - 50), (w, h), color, -1)
    cv2.putText(frame, label, (15, h - 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, _BLACK, 2, cv2.LINE_AA)
    score_txt = (
        f"Liveness: {result.liveness_score:.2f}  "
        f"Identity: {result.identity_score:.2f}  "
        f"Combined: {result.combined_score:.2f}"
    )
    cv2.putText(frame, score_txt, (15, h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _BLACK, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log)
    logger = logging.getLogger(__name__)

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    DB = "embeddings.db"

    # ── Pipeline components ───────────────────────────────────────────
    detector   = build_detector(args)
    detector.warmup()
    tracker    = ByteTracker(track_thresh=0.5)
    aligner    = LandmarkFreeAligner(output_size=(112, 112))

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    rgb_ext    = RGBFeatureExtractor(pretrained=False).to(device).eval()
    flow_ext   = OpticalFlowExtractor()
    mot_enc    = MotionEncoder().to(device).eval()
    tex_ext    = TextureFeatureExtractor()
    tex_enc    = TextureEncoder().to(device).eval()
    fusion     = TemporalMultiModalFusionTransformer().to(device).eval()
    live_head  = LivenessHead().to(device).eval()
    live_eval  = LivenessEvaluator()
    verifier   = ArcFaceVerifier(db_path=DB, verification_threshold=args.id_thr)
    verifier._load()
    engine     = AuthDecisionEngine(
        liveness_threshold=args.live_thr,
        identity_threshold=args.id_thr,
        db_path=DB,
    )

    # State
    face_seqs  : dict[int, deque[np.ndarray]] = {}
    flow_hists : dict[int, deque[np.ndarray]] = {}
    last_result: AuthResult | None = None

    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    WIN = "VisionAuth — Full Auth Pipeline  [E:Enroll | A:AuditLog | C:Clear | Q:Exit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.width, args.height)

    logger.info("Pipeline ready. Press E to enroll, Q to quit.")

    while True:
        packet = cap.read()
        if packet is None:
            time.sleep(0.005)
            continue

        # ── Detect & Track ────────────────────────────────────────────
        raw_faces     = detector.detect(packet.frame,
                                        frame_index=packet.frame_index,
                                        timestamp=packet.timestamp)
        tracked_faces = tracker.update(raw_faces)
        primary       = select_primary_face(tracked_faces)

        display = packet.frame.copy()
        H, W    = display.shape[:2]

        # Prune dead tracks
        active_ids = {f.track_id for f in tracked_faces}
        face_seqs  = {k: v for k, v in face_seqs.items()  if k in active_ids}
        flow_hists = {k: v for k, v in flow_hists.items() if k in active_ids}

        if primary is not None:
            tid = primary.track_id
            curr = aligner.align(packet.frame, primary.bbox)

            if tid not in face_seqs:
                face_seqs[tid]  = deque(maxlen=args.seq_len)
                flow_hists[tid] = deque(maxlen=args.seq_len)
            face_seqs[tid].append(curr)

            x1, y1, x2, y2 = map(int, primary.bbox)
            seq_size = len(face_seqs[tid])

            if seq_size >= args.seq_len:
                frames = list(face_seqs[tid])
                rgb_t, flow_t, tex_t = [], [], []

                for i, f in enumerate(frames):
                    prev = frames[i - 1] if i > 0 else None

                    f_rgb  = (torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                              ).unsqueeze(0).to(device)
                    flow   = flow_ext.compute_flow(f, prev)
                    f_flow = flow_ext.preprocess_flow(flow).to(device)
                    t_map  = tex_ext.extract_texture_maps(f)
                    f_tex  = tex_ext.preprocess_texture(t_map).to(device)

                    if i == args.seq_len - 1:
                        flow_hists[tid].append(flow)

                    with torch.no_grad():
                        rgb_t.append(rgb_ext(f_rgb))
                        flow_t.append(mot_enc(f_flow))
                        tex_t.append(tex_enc(f_tex))

                rgb_seq  = torch.stack(rgb_t,  dim=1)
                flow_seq = torch.stack(flow_t, dim=1)
                tex_seq  = torch.stack(tex_t,  dim=1)

                with torch.no_grad():
                    fused      = fusion(rgb_seq, flow_seq, tex_seq)
                    live_score = float(live_head(fused).cpu().numpy()[0, 0])

                tex_var         = LivenessHeuristics.analyze_texture(curr)
                _, flow_var     = LivenessHeuristics.analyze_motion(list(flow_hists[tid]))
                final_live, status = live_eval.evaluate(live_score, tex_var, flow_var)

                # Identity: 1:N search
                id_user, id_score = verifier.identify_user(packet.frame)
                claimed = id_user if id_user else "UNKNOWN"

                # Decision
                result = engine.decide(
                    username_claimed=claimed,
                    liveness_score=final_live,
                    identity_score=id_score,
                    liveness_status=status,
                )
                last_result = result

                box_color = _GREEN if result.granted else _RED
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(display,
                            f"{'✓ ' if result.granted else '✗ '}{claimed}",
                            (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(display, (x1, y1), (x2, y2), _YELLOW, 2)
                cv2.putText(display,
                            f"Buffering {seq_size}/{args.seq_len}",
                            (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _YELLOW, 1, cv2.LINE_AA)
        else:
            last_result = None

        # Draw banner
        if last_result is not None:
            draw_banner(display, last_result, H, W)

        # HUD hint
        cv2.putText(display,
                    "E: Enroll   A: Audit Log   C: Clear   Q: Quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("e"):
            cap.pause()
            print("\n" + "=" * 52)
            uname = input("  Enter username to enroll: ").strip()
            print("=" * 52)
            if uname:
                cap.resume()
                time.sleep(0.4)
                pkt = cap.read()
                if pkt is not None:
                    ok, msg = verifier.enroll_user(uname, pkt.frame)
                    print(f"  >> {msg}\n")
                else:
                    print("  >> Could not grab frame.\n")
            else:
                print("  >> Cancelled (empty username).\n")
                cap.resume()
        elif key == ord("a"):
            records = engine.audit.recent(10)
            print("\n" + "─" * 60)
            print(f"{'TS':26}  {'User':15}  {'Decision':8}  L/I Scores")
            print("─" * 60)
            for r in records:
                print(f"{r.timestamp:26}  {r.username_claimed:15}  "
                      f"{r.decision:8}  {r.liveness_score:.2f}/{r.identity_score:.2f}")
            print("─" * 60 + "\n")
        elif key == ord("c"):
            verifier.db.clear_database()
            engine.audit.clear()
            last_result = None
            print("  >> Database and audit log cleared.\n")

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 9 demo finished.")


if __name__ == "__main__":
    main()
