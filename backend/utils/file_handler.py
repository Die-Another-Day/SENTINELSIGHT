import os
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
from fastapi import UploadFile
from loguru import logger

from config import settings


class FileHandler:
    """
    SentinelSight File Handler

    Responsible for:
    - Upload validation & storage
    - Remote image downloads
    - File cleanup & metadata
    """

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Validation
    # =====================================================
    def validate_file(self, filename: str) -> bool:
        """
        Validate file extension against allowed types
        """
        extension = Path(filename).suffix.lower()
        return extension in settings.ALLOWED_EXTENSIONS

    # =====================================================
    # Upload Handling
    # =====================================================
    async def save_upload(self, file: UploadFile) -> str:
        """
        Persist uploaded file to disk
        """
        try:
            file_id = uuid.uuid4().hex
            extension = Path(file.filename).suffix
            filename = f"{file_id}{extension}"
            file_path = self.upload_dir / filename

            async with aiofiles.open(file_path, "wb") as buffer:
                content = await file.read()
                await buffer.write(content)

            logger.info(f"[SentinelSight] Upload saved: {file_path}")
            return str(file_path)

        except Exception as exc:
            logger.error(f"[SentinelSight] Upload save failed: {exc}")
            raise

    # =====================================================
    # Remote Image Download
    # =====================================================
    async def download_image(self, url: str) -> str:
        """
        Download image from a remote URL
        """
        try:
            file_id = uuid.uuid4().hex
            filename = f"{file_id}.jpg"
            file_path = self.upload_dir / filename

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(response.content)

            logger.info(f"[SentinelSight] Image downloaded: {url}")
            return str(file_path)

        except Exception as exc:
            logger.error(f"[SentinelSight] Image download failed: {exc}")
            raise

    # =====================================================
    # Cleanup & Utilities
    # =====================================================
    async def cleanup_file(self, file_path: str) -> None:
        """
        Remove file from filesystem
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[SentinelSight] File cleaned: {file_path}")
        except Exception as exc:
            logger.error(f"[SentinelSight] Cleanup failed: {exc}")

    def get_file_size(self, file_path: str) -> int:
        """
        Return file size in bytes
        """
        return os.path.getsize(file_path)

    def is_valid_path(self, path: str) -> bool:
        """
        Verify file path exists and is readable
        """
        return os.path.exists(path) and os.access(path, os.R_OK)
