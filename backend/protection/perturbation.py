import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from loguru import logger

from config import settings


class ImageProtector:
    """
    SentinelSight Image Protection Engine

    Applies adversarial perturbations to images to reduce the
    effectiveness of deepfake, deepnude, and facial misuse models.

    Supported methods:
    - fawkes
    - adversarial
    - lowkey
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[SentinelSight] ImageProtector running on {self.device}")

    # =====================================================
    # Public API
    # =====================================================
    async def protect_image(
        self,
        image_path: str,
        strength: Optional[float] = None,
        method: str = "fawkes"
    ) -> str:
        """
        Apply protection method to an image and save output
        """
        try:
            strength = strength if strength is not None else settings.PERTURBATION_STRENGTH
            logger.info(
                f"[SentinelSight] Protecting image | method={method}, strength={strength}"
            )

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if method == "fawkes":
                protected = self._fawkes_protection(image, strength)
            elif method == "adversarial":
                protected = self._adversarial_protection(image, strength)
            elif method == "lowkey":
                protected = self._lowkey_protection(image, strength)
            else:
                raise ValueError(f"Unsupported protection method: {method}")

            output_path = self._get_output_path(image_path, method)
            protected_bgr = cv2.cvtColor(protected, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, protected_bgr)

            logger.info(f"[SentinelSight] Protected image saved → {output_path}")
            return output_path

        except Exception as exc:
            logger.error(f"[SentinelSight] Image protection failed: {exc}")
            raise

    # =====================================================
    # Protection Methods
    # =====================================================
    def _fawkes_protection(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Frequency-domain perturbation inspired by Fawkes-style cloaking
        """
        try:
            img = image.astype(np.float32) / 255.0
            protected = np.zeros_like(img)

            rows, cols = img.shape[:2]
            crow, ccol = rows // 2, cols // 2

            r, c = np.ogrid[:rows, :cols]
            distance = np.sqrt((r - crow) ** 2 + (c - ccol) ** 2)

            mid_freq_mask = np.exp(-((distance - 50) ** 2) / (2 * 30 ** 2))
            mid_freq_mask /= mid_freq_mask.max() + 1e-8

            for channel in range(3):
                fft = np.fft.fftshift(np.fft.fft2(img[:, :, channel]))
                magnitude = np.abs(fft)
                phase = np.angle(fft)

                noise = np.random.randn(rows, cols) * strength * 0.01
                magnitude *= (1 + noise * mid_freq_mask)

                perturbed = magnitude * np.exp(1j * phase)
                restored = np.fft.ifft2(np.fft.ifftshift(perturbed))
                protected[:, :, channel] = np.real(restored)

            protected = np.clip(protected, 0, 1)
            return (protected * 255).astype(np.uint8)

        except Exception as exc:
            logger.error(f"[SentinelSight] Fawkes protection error: {exc}")
            return image

    def _adversarial_protection(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Gradient-weighted adversarial noise injection
        """
        try:
            img = image.astype(np.float32) / 255.0

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient /= gradient.max() + 1e-8

            perturbation = np.random.randn(*img.shape) * strength * 0.02
            perturbation *= gradient[:, :, None]

            protected = img + perturbation
            protected = np.clip(protected, 0, 1)

            return (protected * 255).astype(np.uint8)

        except Exception as exc:
            logger.error(f"[SentinelSight] Adversarial protection error: {exc}")
            return image

    def _lowkey_protection(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Subtle high-frequency noise inspired by LowKey-style cloaking
        """
        try:
            img = image.astype(np.float32) / 255.0
            h, w = img.shape[:2]

            noise = np.zeros_like(img, dtype=np.float32)

            for c in range(3):
                base = np.random.randn(h // 4, w // 4)
                upsampled = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
                blurred = cv2.GaussianBlur(upsampled, (5, 5), 0)
                noise[:, :, c] = upsampled - blurred

            noise /= np.std(noise) + 1e-8
            protected = img + noise * strength * 0.015

            protected = np.clip(protected, 0, 1)
            return (protected * 255).astype(np.uint8)

        except Exception as exc:
            logger.error(f"[SentinelSight] LowKey protection error: {exc}")
            return image

    # =====================================================
    # Utilities
    # =====================================================
    def _get_output_path(self, input_path: str, method: str) -> str:
        """
        Generate output path for protected images
        """
        src = Path(input_path)
        output_dir = Path(settings.PROCESSED_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{src.stem}_protected_{method}{src.suffix}"
        return str(output_dir / filename)


# =========================================================
# Face-Specific Protection
# =========================================================
class FaceProtector:
    """
    SentinelSight Face-Level Protection

    Applies perturbations only to detected face regions.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def protect_faces(
        self,
        image: np.ndarray,
        strength: float = 0.05
    ) -> np.ndarray:
        """
        Apply perturbation selectively to detected faces
        """
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            protected = image.copy()

            for x, y, w, h in faces:
                face = image[y:y + h, x:x + w].astype(np.float32) / 255.0
                noise = np.random.randn(*face.shape) * strength

                face_protected = np.clip(face + noise, 0, 1)
                protected[y:y + h, x:x + w] = (face_protected * 255).astype(np.uint8)

            return protected

        except Exception as exc:
            logger.error(f"[SentinelSight] Face protection error: {exc}")
            return image
