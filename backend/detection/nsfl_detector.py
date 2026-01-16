import numpy as np
import cv2
from typing import Dict
from loguru import logger


class NSFLDetector:
    """
    NSFL (Not Safe For Life) detector
    Targets gore, extreme violence, disturbing imagery
    """

    def __init__(self):
        logger.info("NSFL detector initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect NSFL content in an RGB image
        """
        try:
            techniques = {
                "blood_presence": self._detect_blood_colors(image),
                "violence_indicators": self._detect_violence_indicators(image),
                "disturbing_patterns": self._detect_disturbing_patterns(image),
                "medical_gore_context": self._detect_medical_content(image)
            }

            # Weighted conservative aggregation
            blood = techniques["blood_presence"]
            violence = techniques["violence_indicators"]
            disturbing = techniques["disturbing_patterns"]
            medical = techniques["medical_gore_context"]

            # Blood is strongest signal, but not absolute alone
            combined_score = max(
                blood * 1.25,
                (blood + violence) / 2.0,
                (disturbing + violence) / 2.2,
                medical
            )

            final_score = float(np.clip(combined_score, 0.0, 1.0))

            return {
                "confidence": final_score,
                "details": {
                    "techniques": techniques,
                    "aggregation": "conservative_weighted_max",
                    "model": "custom_nsfl_forensic"
                }
            }

        except Exception as e:
            logger.error(f"NSFL detection error: {e}")
            return {"confidence": 0.0, "details": {"error": str(e)}}

    # ------------------------------------------------------------------
    # Detection Techniques
    # ------------------------------------------------------------------
    def _detect_blood_colors(self, image: np.ndarray) -> float:
        """Detect blood-like dark red regions"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Red color ranges
            red_masks = [
                cv2.inRange(hsv, (0, 50, 40), (10, 255, 255)),
                cv2.inRange(hsv, (170, 50, 40), (180, 255, 255))
            ]
            red_mask = cv2.bitwise_or(*red_masks)

            red_ratio = np.mean(red_mask > 0)

            # Dark red = more blood-like
            dark_red_mask = cv2.inRange(
                hsv, (0, 100, 20), (10, 255, 120)
            )
            dark_ratio = np.mean(dark_red_mask > 0)

            score = (red_ratio * 1.8) + (dark_ratio * 2.8)
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0

    def _detect_violence_indicators(self, image: np.ndarray) -> float:
        """Detect sharp weapons / violent structures"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 160)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=70,
                minLineLength=60,
                maxLineGap=8
            )

            line_score = 0.0
            if lines is not None:
                line_density = len(lines) / (image.shape[0] * image.shape[1] / 10000)
                line_score = min(line_density / 3.5, 1.0)

            contrast = np.std(gray)
            contrast_score = min(max(contrast - 50, 0) / 90.0, 1.0)

            return max(line_score, contrast_score) * 0.6

        except Exception:
            return 0.0

    def _detect_disturbing_patterns(self, image: np.ndarray) -> float:
        """Detect chaotic or extreme visual distributions"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            hist = hist / (hist.sum() + 1e-6)
            hist = hist[hist > 0]

            entropy = -np.sum(hist * np.log2(hist))

            entropy_score = 0.0
            if entropy < 3.2 or entropy > 7.6:
                entropy_score = min(abs(entropy - 5.4) / 3.0, 1.0)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            chaos_score = min(np.var(lap) / 12000.0, 1.0)

            return max(entropy_score, chaos_score) * 0.7

        except Exception:
            return 0.0

    def _detect_medical_content(self, image: np.ndarray) -> float:
        """Detect surgical gore (not general hospitals)"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]

            bright_ratio = np.mean(l > 200)

            if bright_ratio < 0.25:
                return 0.0

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            flesh_mask = cv2.inRange(
                hsv, (0, 20, 70), (20, 255, 255)
            )
            flesh_ratio = np.mean(flesh_mask > 0)

            if flesh_ratio > 0.18 and bright_ratio > 0.35:
                return float(np.clip(flesh_ratio * bright_ratio * 4.0, 0.0, 1.0))

            return 0.0

        except Exception:
            return 0.0
