import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple
from loguru import logger

try:
    from facenet_pytorch import MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logger.warning("FaceNet not available, falling back to OpenCV Haar Cascade")


class DeepfakeDetector:
    """
    Deepfake detector for face manipulation artifacts.
    Uses forensic heuristics instead of black-box classification.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = None
        self._load_models()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        try:
            if FACENET_AVAILABLE:
                logger.info("Loading MTCNN face detector")
                self.face_detector = MTCNN(
                    keep_all=True,
                    device=self.device,
                    thresholds=[0.6, 0.7, 0.7]
                )
            else:
                logger.info("Loading OpenCV Haar Cascade face detector")
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.face_detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect deepfake manipulation in an RGB image.
        """
        try:
            faces = self._detect_faces(image)

            if not faces:
                return {
                    "confidence": 0.0,
                    "details": {
                        "faces_detected": 0,
                        "message": "No faces detected"
                    }
                }

            face_results = []
            scores = []

            for face_region in faces:
                analysis = self._analyze_face(face_region, image)
                face_results.append(analysis)
                scores.append(analysis["score"])

            final_score = float(np.clip(max(scores), 0.0, 1.0))

            return {
                "confidence": final_score,
                "details": {
                    "faces_detected": len(faces),
                    "face_analyses": face_results,
                    "aggregation": "max-face-score",
                    "model": "mtcnn" if FACENET_AVAILABLE else "opencv_haar"
                }
            }

        except Exception as e:
            logger.error(f"Deepfake detection failed: {e}")
            return {"confidence": 0.0, "details": {"error": str(e)}}

    # ------------------------------------------------------------------
    # Face Detection
    # ------------------------------------------------------------------
    def _detect_faces(self, image: np.ndarray) -> List[Tuple]:
        try:
            if FACENET_AVAILABLE and self.face_detector:
                from PIL import Image as PILImage
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                pil_img = PILImage.fromarray(image)
                boxes, probs = self.face_detector.detect(pil_img)

                if boxes is None:
                    return []

                return [(box, float(prob)) for box, prob in zip(boxes, probs)]

            # OpenCV fallback
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            return [(face, 1.0) for face in faces]

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    # ------------------------------------------------------------------
    # Face Analysis
    # ------------------------------------------------------------------
    def _analyze_face(self, face_region: Tuple, image: np.ndarray) -> Dict:
        try:
            box, face_conf = face_region

            if FACENET_AVAILABLE:
                x1, y1, x2, y2 = map(int, box)
            else:
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                return {"score": 0.0, "techniques": {}}

            # Reject extremely small faces
            if face.shape[0] < 40 or face.shape[1] < 40:
                return {"score": 0.0, "techniques": {"face_quality": "too_small"}}

            techniques = {
                "frequency_artifacts": self._frequency_artifacts(face),
                "facial_symmetry": self._facial_symmetry(face),
                "lighting_inconsistency": self._lighting_inconsistency(face),
                "edge_artifacts": self._edge_artifacts(face)
            }

            raw_score = np.mean(list(techniques.values()))
            weighted_score = raw_score * face_conf

            return {
                "score": float(np.clip(weighted_score, 0.0, 1.0)),
                "face_confidence": round(face_conf, 3),
                "techniques": techniques
            }

        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            return {"score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Detection Techniques
    # ------------------------------------------------------------------
    def _frequency_artifacts(self, face: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            dct = cv2.dct(np.float32(gray))

            h, w = dct.shape
            high_freq = dct[h // 2 :, w // 2 :]
            ratio = np.sum(np.abs(high_freq)) / (np.sum(np.abs(dct)) + 1e-6)

            return np.clip(abs(ratio - 0.15) * 6, 0.0, 1.0)

        except Exception:
            return 0.0

    def _facial_symmetry(self, face: np.ndarray) -> float:
        try:
            h, w = face.shape[:2]
            left = face[:, : w // 2]
            right = cv2.flip(face[:, w // 2 :], 1)

            min_w = min(left.shape[1], right.shape[1])
            diff = np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))

            return np.clip(diff / 120.0, 0.0, 1.0)

        except Exception:
            return 0.0

    def _lighting_inconsistency(self, face: np.ndarray) -> float:
        try:
            lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
            l = lab[:, :, 0]

            gx = cv2.Sobel(l, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(l, cv2.CV_64F, 0, 1)

            grad_std = np.std(np.sqrt(gx**2 + gy**2))
            return np.clip(grad_std / 45.0, 0.0, 1.0)

        except Exception:
            return 0.0

    def _edge_artifacts(self, face: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            h, w = edges.shape
            bw = max(4, min(h, w) // 12)

            border = np.concatenate([
                edges[:bw, :].flatten(),
                edges[-bw:, :].flatten(),
                edges[:, :bw].flatten(),
                edges[:, -bw:].flatten()
            ])

            density = np.mean(border) / 255.0
            return np.clip(density * 1.8, 0.0, 1.0)

        except Exception:
            return 0.0
