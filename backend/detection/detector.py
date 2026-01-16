import asyncio
import cv2
import numpy as np
from datetime import datetime
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from config import settings
from detection.nsfw_detector import NSFWDetector
from detection.deepfake_detector import DeepfakeDetector
from detection.deepnude_detector import DeepnudeDetector
from detection.nsfl_detector import NSFLDetector
from api_integrations.ai_apis import AIAPIsIntegration


class ImageDetector:
    """
    SentinelSight Unified Image Detection Engine

    Coordinates multiple internal ML detectors and optional
    external AI APIs to produce a consolidated verdict.
    """

    def __init__(self):
        self.models_loaded = False 
        self.executor = ThreadPoolExecutor(
            max_workers=settings.API_WORKERS or 4
        )

        logger.info("[SentinelSight] Initializing detection modules...")

        try:
            self.nsfw_detector = NSFWDetector()
            self.deepfake_detector = DeepfakeDetector()
            self.deepnude_detector = DeepnudeDetector()
            self.nsfl_detector = NSFLDetector()
            self.ai_apis = AIAPIsIntegration()
            
            self.models_loaded = True

            logger.success("[SentinelSight] Detection modules ready")

        except Exception as exc:
            logger.critical(f"[SentinelSight] Model initialization failed: {exc}")
            self.models_loaded = False
            raise

    # =====================================================
    # Public API
    # =====================================================
    async def analyze_image(
        self,
        image_path: str,
        use_external_apis: bool = False
    ) -> Dict:
        """
        Perform full detection pipeline on a single image
        """
        try:
            image = await self._load_image(image_path)
            logger.info(f"[SentinelSight] Analyzing image → {image_path}")

            tasks = [
                self._run(self.nsfw_detector.detect, image),
                self._run(self.deepfake_detector.detect, image),
                self._run(self.deepnude_detector.detect, image),
                self._run(self.nsfl_detector.detect, image),
            ]

            if use_external_apis:
                tasks.append(self._run_external_api_detection(image_path))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            return self._aggregate_results(
                image_path=image_path,
                nsfw_result=self._safe_result(results, 0),
                deepfake_result=self._safe_result(results, 1),
                deepnude_result=self._safe_result(results, 2),
                nsfl_result=self._safe_result(results, 3),
                external_api_result=self._safe_result(results, 4),
            )

        except Exception as exc:
            logger.error(f"[SentinelSight] Image analysis failed: {exc}")
            raise

    # =====================================================
    # Internal Helpers
    # =====================================================
    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, fn, *args)

    async def _load_image(self, image_path: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._load_image_sync, image_path
        )

    @staticmethod
    def _load_image_sync(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    async def _run_external_api_detection(self, image_path: str) -> Dict:
        return await self.ai_apis.analyze_image(image_path)

    @staticmethod
    def _safe_result(results, index: int) -> Dict:
        """
        Gracefully extract async result
        """
        if index >= len(results):
            return {}
        result = results[index]
        return {} if isinstance(result, Exception) else result

    # =====================================================
    # Aggregation Logic
    # =====================================================
    def _aggregate_results(
        self,
        image_path: str,
        nsfw_result: Dict,
        deepfake_result: Dict,
        deepnude_result: Dict,
        nsfl_result: Dict,
        external_api_result: Dict
    ) -> Dict:
        """
        Combine all detector outputs into a final verdict
        """

        scores = {
            "nsfw": float(nsfw_result.get("confidence", 0.0)),
            "deepfake": float(deepfake_result.get("confidence", 0.0)),
            "deepnude": float(deepnude_result.get("confidence", 0.0)),
            "nsfl": float(nsfl_result.get("confidence", 0.0)),
        }

        flags = {
            "is_nsfw": scores["nsfw"] >= settings.NSFW_THRESHOLD,
            "is_deepfake": scores["deepfake"] >= settings.DEEPFAKE_THRESHOLD,
            "is_deepnude": scores["deepnude"] >= settings.DEEPNUDE_THRESHOLD,
            "is_nsfl": scores["nsfl"] >= 0.75,
        }

        warnings = [
            f"{key.replace('_', ' ').upper()} detected ({scores[key]:.2%})"
            for key, triggered in flags.items()
            if triggered
        ]

        result = {
            "filename": image_path.split("/")[-1],
            **flags,
            "confidence_scores": {
                k: round(v, 4) for k, v in scores.items()
            },
            "details": {
                "nsfw": nsfw_result.get("details", {}),
                "deepfake": deepfake_result.get("details", {}),
                "deepnude": deepnude_result.get("details", {}),
                "nsfl": nsfl_result.get("details", {}),
                "external_apis": external_api_result.get("details", {}),
            },
            "warnings": warnings,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if external_api_result:
            result["confidence_scores"]["external_apis"] = (
                external_api_result.get("scores", {})
            )

        logger.info(
            f"[SentinelSight] Verdict → NSFW={flags['is_nsfw']}, "
            f"Deepfake={flags['is_deepfake']}, "
            f"Deepnude={flags['is_deepnude']}, "
            f"NSFL={flags['is_nsfl']}"
        )

        return result

    # =====================================================
    # Cleanup
    # =====================================================
    def shutdown(self):
        logger.info("[SentinelSight] Shutting down detection executor")
        self.executor.shutdown(wait=True)

    def __del__(self):
        self.shutdown()
