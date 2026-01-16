import numpy as np
import cv2
from typing import Dict, Any
from loguru import logger


class DeepnudeDetector:
    """
    Deepnude-specific detector for SentinelSight
    ------------------------------------------------
    Detects AI-generated explicit content by analyzing:
    - GAN frequency artifacts
    - Texture inconsistencies
    - Color bleeding
    - Unnatural skin distributions
    - Generation boundary artifacts
    """

    def __init__(self, resize_max: int = 1024):
        self.resize_max = resize_max
        logger.info("DeepnudeDetector initialized")

        # Weighting based on reliability of each signal
        self.weights = {
            "gan_artifacts": 0.25,
            "texture_inconsistency": 0.20,
            "color_bleeding": 0.15,
            "unnatural_skin": 0.20,
            "generation_boundaries": 0.20
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run deepnude detection on an image.

        Args:
            image (np.ndarray): RGB image

        Returns:
            Dict containing confidence score and technique breakdown
        """
        try:
            image = self._preprocess_image(image)

            techniques = {
                "gan_artifacts": self._detect_gan_artifacts(image),
                "texture_inconsistency": self._detect_texture_inconsistency(image),
                "color_bleeding": self._detect_color_bleeding(image),
                "unnatural_skin": self._detect_unnatural_skin(image),
                "generation_boundaries": self._detect_generation_boundaries(image),
            }

            # Weighted aggregation (better than simple mean)
            weighted_score = sum(
                techniques[k] * self.weights[k] for k in techniques
            )

            logger.debug(f"Deepnude technique scores: {techniques}")

            return {
                "confidence": float(np.clip(weighted_score, 0.0, 1.0)),
                "details": {
                    "techniques": techniques,
                    "weights": self.weights,
                    "model": "sentinelsight_deepnude_v1"
                }
            }

        except Exception as e:
            logger.exception("Deepnude detection failed")
            return {
                "confidence": 0.0,
                "details": {"error": str(e)}
            }

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Validate and resize image safely"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB image")

        h, w, _ = image.shape
        max_dim = max(h, w)

        if max_dim > self.resize_max:
            scale = self.resize_max / max_dim
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        return image

    # ------------------------------------------------------------------
    # Detection Techniques
    # ------------------------------------------------------------------
    def _detect_gan_artifacts(self, image: np.ndarray) -> float:
        """Detect GAN frequency-domain artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fshift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = np.log(np.abs(fshift) + 1)

            h, w = magnitude.shape
            center = magnitude[h//4:3*h//4, w//4:3*w//4]
            outer = magnitude.copy()
            outer[h//4:3*h//4, w//4:3*w//4] = 0

            ratio = np.mean(center) / (np.mean(outer[outer > 0]) + 1e-6)

            score = abs(np.log10(ratio))
            return float(np.clip(score / 2.0, 0.0, 1.0))

        except Exception as e:
            logger.error(f"GAN artifact detection error: {e}")
            return 0.0

    def _detect_texture_inconsistency(self, image: np.ndarray) -> float:
        """Detect unnatural texture variance patterns"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)

            # Sliding window variance
            variances = []
            block = 32
            for y in range(0, gray.shape[0] - block, block):
                for x in range(0, gray.shape[1] - block, block):
                    variances.append(np.var(lap[y:y+block, x:x+block]))

            if not variances:
                return 0.0

            return float(np.clip(np.var(variances) / 1e4, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Texture inconsistency error: {e}")
            return 0.0

    def _detect_color_bleeding(self, image: np.ndarray) -> float:
        """Detect abnormal chroma transitions"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            a, b = lab[:, :, 1], lab[:, :, 2]

            grad_a = cv2.Laplacian(a, cv2.CV_64F)
            grad_b = cv2.Laplacian(b, cv2.CV_64F)

            score = (np.percentile(np.abs(grad_a), 95) +
                     np.percentile(np.abs(grad_b), 95)) / 2

            return float(np.clip(score / 50.0, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Color bleeding error: {e}")
            return 0.0

    def _detect_unnatural_skin(self, image: np.ndarray) -> float:
        """Detect statistically unnatural skin tone distributions"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            lower = np.array([0, 20, 70], np.uint8)
            upper = np.array([20, 255, 255], np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            if np.count_nonzero(mask) < 500:
                return 0.0

            skin_pixels = image[mask > 0]
            std = np.mean(np.std(skin_pixels, axis=0))

            deviation = abs(std - 25.0) / 25.0
            return float(np.clip(deviation, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Skin analysis error: {e}")
            return 0.0

    def _detect_generation_boundaries(self, image: np.ndarray) -> float:
        """Detect abrupt boundary transitions from synthesis blending"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            smooth = cv2.bilateralFilter(gray, 9, 75, 75)

            diff = np.abs(gray.astype(np.float32) - smooth.astype(np.float32))
            edge_density = np.mean(diff > np.percentile(diff, 90))

            return float(np.clip(edge_density * 5.0, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Boundary detection error: {e}")
            return 0.0
