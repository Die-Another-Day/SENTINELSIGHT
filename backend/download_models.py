#!/usr/bin/env python3
"""
SentinelSight – Model Setup Utility
Handles initial directory setup and optional model warm-up
"""

import sys
from pathlib import Path
from loguru import logger


# =====================================================
# Optional Model Downloads
# =====================================================
def download_nudenet():
    """
    NudeNet models auto-download on first inference.
    This function exists for visibility and validation.
    """
    try:
        logger.info("[SentinelSight] NudeNet models will auto-download on first use")
        return True
    except Exception as e:
        logger.error(f"[SentinelSight] NudeNet error: {str(e)}")
        return False


def download_facenet():
    """
    Preloads FaceNet models (optional)
    Useful for local development, skipped safely in production
    """
    try:
        logger.info("[SentinelSight] Initializing FaceNet models...")

        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MTCNN(keep_all=True, device=device)
        logger.info("[SentinelSight] MTCNN ready")

        InceptionResnetV1(pretrained="vggface2").eval().to(device)
        logger.info("[SentinelSight] InceptionResnetV1 ready")

        return True

    except Exception as e:
        logger.warning("[SentinelSight] FaceNet optional dependency not available")
        logger.debug(str(e))
        return True


# =====================================================
# Directory Setup
# =====================================================
def setup_directories():
    """Create required runtime directories"""
    directories = [
        "models",
        "data/uploads",
        "data/processed",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"[SentinelSight] Directory ready: {directory}")


# =====================================================
# Dependency Check
# =====================================================
def check_dependencies():
    required_packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn"
    }

    missing = []

    for pkg, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"[SentinelSight] Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False

    logger.info("[SentinelSight] All dependencies satisfied")
    return True


# =====================================================
# Main Entry
# =====================================================
def main():
    logger.info("[SentinelSight] Initializing environment setup")

    if not check_dependencies():
        sys.exit(1)

    setup_directories()

    logger.info("[SentinelSight] Model preparation started")
    download_nudenet()
    download_facenet()

    logger.info("[SentinelSight] Setup complete")
    logger.info("Run backend with: python main.py")


if __name__ == "__main__":
    main()
