import os
import asyncio
import uuid
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from loguru import logger

from config import settings
from detection.detector import ImageDetector


class FolderScanner:
    """
    SentinelSight Folder Scanner

    Scans directories for NSFW, deepfake, deepnude, and unsafe content
    using internal detection pipelines.
    """

    def __init__(self):
        self.active_scans: Dict[str, Dict] = {}
        self.detector = ImageDetector()

    # =====================================================
    # Public API
    # =====================================================
    async def start_scan(
        self,
        folder_path: str,
        recursive: bool = True,
        use_external_apis: bool = False
    ) -> str:
        """
        Start an asynchronous folder scan.

        Args:
            folder_path: Target directory
            recursive: Scan subdirectories
            use_external_apis: Enable third-party APIs

        Returns:
            scan_id (str)
        """
        try:
            scan_id = uuid.uuid4().hex

            self.active_scans[scan_id] = {
                "status": "running",
                "folder_path": folder_path,
                "total_files": 0,
                "processed_files": 0,
                "flagged_files": [],
                "start_time": datetime.utcnow().isoformat(),
                "end_time": None,
                "errors": []
            }

            asyncio.create_task(
                self._run_scan(
                    scan_id=scan_id,
                    folder_path=folder_path,
                    recursive=recursive,
                    use_external_apis=use_external_apis
                )
            )

            logger.info(f"[SentinelSight] Scan started | ID={scan_id}")
            return scan_id

        except Exception as exc:
            logger.exception("[SentinelSight] Failed to start scan")
            raise exc

    # =====================================================
    # Internal Scan Logic
    # =====================================================
    async def _run_scan(
        self,
        scan_id: str,
        folder_path: str,
        recursive: bool,
        use_external_apis: bool
    ):
        """
        Execute scan logic in background task
        """
        try:
            image_files = self._collect_image_files(folder_path, recursive)
            self.active_scans[scan_id]["total_files"] = len(image_files)

            logger.info(
                f"[SentinelSight] Scan {scan_id} | Images found: {len(image_files)}"
            )

            for image_path in image_files:
                try:
                    result = await self.detector.analyze_image(
                        image_path,
                        use_external_apis=use_external_apis
                    )

                    if (
                        result.get("is_nsfw")
                        or result.get("is_deepfake")
                        or result.get("is_deepnude")
                        or result.get("is_nsfl")
                    ):
                        self.active_scans[scan_id]["flagged_files"].append({
                            "file_path": image_path,
                            "result": result
                        })

                    self.active_scans[scan_id]["processed_files"] += 1

                except Exception as file_exc:
                    logger.error(
                        f"[SentinelSight] File scan failed: {image_path}"
                    )
                    self.active_scans[scan_id]["errors"].append({
                        "file": image_path,
                        "error": str(file_exc)
                    })

            self.active_scans[scan_id]["status"] = "completed"
            self.active_scans[scan_id]["end_time"] = datetime.utcnow().isoformat()

            logger.info(f"[SentinelSight] Scan completed | ID={scan_id}")

        except Exception as scan_exc:
            logger.exception(f"[SentinelSight] Scan failed | ID={scan_id}")
            self.active_scans[scan_id]["status"] = "failed"
            self.active_scans[scan_id]["errors"].append({
                "error": str(scan_exc)
            })

    # =====================================================
    # File Collection
    # =====================================================
    def _collect_image_files(
        self,
        folder_path: str,
        recursive: bool
    ) -> List[str]:
        """
        Collect supported image files from directory
        """
        supported_extensions = {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"
        }

        image_files: List[str] = []

        if recursive:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in supported_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file)
                if (
                    os.path.isfile(full_path)
                    and Path(file).suffix.lower() in supported_extensions
                ):
                    image_files.append(full_path)

        return image_files

    # =====================================================
    # Scan Status & Metrics
    # =====================================================
    async def get_scan_status(self, scan_id: str) -> Dict:
        """
        Retrieve scan progress and results
        """
        if scan_id not in self.active_scans:
            raise ValueError("Scan ID not found")

        scan = self.active_scans[scan_id]

        return {
            "scan_id": scan_id,
            "status": scan["status"],
            "folder_path": scan["folder_path"],
            "total_files": scan["total_files"],
            "processed_files": scan["processed_files"],
            "flagged_count": len(scan["flagged_files"]),
            "flagged_files": scan["flagged_files"],
            "start_time": scan["start_time"],
            "end_time": scan["end_time"],
            "errors": scan["errors"]
        }

    async def get_statistics(self) -> Dict:
        """
        Aggregate scanner statistics
        """
        total_scans = len(self.active_scans)

        completed = sum(
            1 for scan in self.active_scans.values()
            if scan["status"] == "completed"
        )

        total_flagged = sum(
            len(scan["flagged_files"])
            for scan in self.active_scans.values()
        )

        return {
            "total_scans": total_scans,
            "completed_scans": completed,
            "running_scans": total_scans - completed,
            "total_flagged_files": total_flagged
        }
