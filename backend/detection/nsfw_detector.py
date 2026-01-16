import os
import tempfile
from typing import Dict

import numpy as np
from loguru import logger

try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
except ImportError:
    NUDENET_AVAILABLE = False
    logger.warning("NudeNet not available – using heuristic fallback")

from config import settings


class NSFWDetector:
    """
    NSFW content detector
    Primary: NudeNet
    Fallback: Skin-tone heuristic
    """

    def __init__(self):
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            if NUDENET_AVAILABLE:
                logger.info("Loading NudeNet NSFW model")
                self.model = NudeDetector()
                logger.success("NudeNet loaded successfully")
            else:
                logger.warning("Running without NudeNet")
        except Exception as e:
            logger.error(f"NSFW model load failed: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Public Detection API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect NSFW content in an RGB image
        """
        try:
            if not self.model or not NUDENET_AVAILABLE:
                return self._basic_detection(image)

            return self._nudenet_detection(image)

        except Exception as e:
            logger.error(f"NSFW detection error: {e}")
            return {"confidence": 0.0, "details": {"error": str(e)}}

    # ------------------------------------------------------------------
    # NudeNet Detection
    # ------------------------------------------------------------------
    def _nudenet_detection(self, image: np.ndarray) -> Dict:
        from PIL import Image

        tmp_path = None

        try:
            # NudeNet requires file path
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray(image).save(tmp_path)

            detections = self.model.detect(tmp_path)

            nsfw_score = 0.0
            detected_classes = {}

            for det in detections:
                class_name = det.get("class")
                score = float(det.get("score", 0.0))

                detected_classes[class_name] = max(
                    detected_classes.get(class_name, 0.0), score
                )

                if class_name in {
                    "FEMALE_GENITALIA_EXPOSED",
                    "MALE_GENITALIA_EXPOSED",
                    "FEMALE_BREAST_EXPOSED",
                    "BUTTOCKS_EXPOSED",
                    "ANUS_EXPOSED"
                }:
                    nsfw_score = max(nsfw_score, score)

            return {
                "confidence": float(min(nsfw_score, 1.0)),
                "details": {
                    "detected_classes": detected_classes,
                    "detection_count": len(detections),
                    "model": "NudeNet"
                }
            }

        except Exception as e:
            logger.error(f"NudeNet detection failed: {e}")
            return {"confidence": 0.0, "details": {"error": str(e)}}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Heuristic Fallback Detection
    # ------------------------------------------------------------------
    def _basic_detection(self, image: np.ndarray) -> Dict:
        """
        Fallback NSFW detection using skin-tone analysis
        ⚠️ Heuristic only – not accurate
        """
        try:
            import cv2

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = float(np.mean(skin_mask > 0))

            confidence = min(skin_ratio * 2.0, 1.0) if skin_ratio > 0.30 else 0.0

            return {
                "confidence": confidence,
                "details": {
                    "skin_ratio": skin_ratio,
                    "model": "heuristic_skin",
                    "note": "Fallback detection – install NudeNet for accuracy"
                }
            }

        except Exception as e:
            logger.error(f"Fallback NSFW detection failed: {e}")
            return {"confidence": 0.0, "details": {"error": str(e)}}
