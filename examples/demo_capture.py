"""
run_feature1.py
===============
Entry point to manually test Feature 1 — Video Capture & Stream.

Usage
-----
    # Default webcam (index 0):
    python run_feature1.py

    # Specific camera index:
    python run_feature1.py --source 1

    # Video file:
    python run_feature1.py --source path/to/video.mp4

    # Custom resolution / FPS:
    python run_feature1.py --width 1280 --height 720 --fps 60
"""

import argparse
import logging

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.capture.preview import run_preview


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 1: Video Capture Demo")
    p.add_argument("--source", default=0,    help="Camera index or video file path (default: 0)")
    p.add_argument("--width",  type=int, default=640,  help="Frame width  (default: 640)")
    p.add_argument("--height", type=int, default=480,  help="Frame height (default: 480)")
    p.add_argument("--fps",    type=float, default=30.0, help="Target FPS  (default: 30)")
    p.add_argument("--buffer", type=int, default=64,   help="Frame buffer size (default: 64)")
    p.add_argument("--log",    default="INFO",          help="Log level: DEBUG|INFO|WARNING|ERROR")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log)
    logger = logging.getLogger(__name__)

    # Convert source to int if it looks like a camera index
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string (file path)

    logger.info("Starting Feature 1 — Video Capture")
    logger.info("Source: %s | %dx%d @ %.0f FPS | Buffer: %d frames",
                source, args.width, args.height, args.fps, args.buffer)

    cap = create_capture(
        source_id   = source,
        width       = args.width,
        height      = args.height,
        target_fps  = args.fps,
        buffer_size = args.buffer,
    )
    cap.start()
    run_preview(cap)   # blocks until window is closed
    logger.info("Feature 1 demo finished.")


if __name__ == "__main__":
    main()
