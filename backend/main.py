from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import uvicorn
import os
from loguru import logger

from config import settings
from detection.detector import ImageDetector
from protection.perturbation import ImageProtector
from utils.file_handler import FileHandler
from utils.scanner import FolderScanner
from utils.directory_browser import DirectoryBrowser

# =====================================================
# SentinelSight – Backend API
# =====================================================

app = FastAPI(
    title="SentinelSight API",
    description="AI-powered Image Safety, Deepfake & DeepNude Detection System",
    version="1.0.0"
)

# =====================================================
# CORS Configuration (Vercel + Local)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Initialize Core Components
# =====================================================
detector = ImageDetector()
protector = ImageProtector()
file_handler = FileHandler()
scanner = FolderScanner()
dir_browser = DirectoryBrowser()

# =====================================================
# Logging
# =====================================================
logger.add(
    settings.LOG_FILE,
    rotation="100 MB",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# =====================================================
# Request / Response Models
# =====================================================
class AnalyzeURLRequest(BaseModel):
    url: HttpUrl
    use_external_apis: bool = False


class AnalyzeFolderRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    use_external_apis: bool = False


class ProtectImageRequest(BaseModel):
    image_path: str
    strength: Optional[float] = None
    method: str = "fawkes"


class DetectionResult(BaseModel):
    filename: str
    is_nsfw: bool
    is_deepfake: bool
    is_deepnude: bool
    is_nsfl: bool
    confidence_scores: dict
    details: dict
    warnings: List[str]
    timestamp: str


# =====================================================
# Health Check
# =====================================================
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "project": "SentinelSight",
        "models_loaded": detector.models_loaded,
        "version": "1.0.0"
    }


# =====================================================
# Image Analysis
# =====================================================
@app.post("/api/analyze", response_model=DetectionResult)
async def analyze_image(
    file: UploadFile = File(...),
    use_external_apis: bool = False
):
    try:
        if not file_handler.validate_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        file_path = await file_handler.save_upload(file)

        logger.info(f"[SentinelSight] Analyzing image: {file.filename}")

        result = await detector.analyze_image(
            file_path,
            use_external_apis=use_external_apis
        )

        await file_handler.cleanup_file(file_path)
        return result

    except Exception as e:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# URL Analysis
# =====================================================
@app.post("/api/analyze-url", response_model=DetectionResult)
async def analyze_url(request: AnalyzeURLRequest):
    try:
        file_path = await file_handler.download_image(str(request.url))

        result = await detector.analyze_image(
            file_path,
            use_external_apis=request.use_external_apis
        )

        await file_handler.cleanup_file(file_path)
        return result

    except Exception as e:
        logger.exception("URL analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Folder Scan
# =====================================================
@app.post("/api/analyze-folder")
async def analyze_folder(
    request: AnalyzeFolderRequest,
    background_tasks: BackgroundTasks
):
    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=400, detail="Path does not exist")

    scan_id = await scanner.start_scan(
        request.folder_path,
        recursive=request.recursive,
        use_external_apis=request.use_external_apis
    )

    return {
        "scan_id": scan_id,
        "status": "scanning",
        "project": "SentinelSight"
    }


@app.get("/api/scan-status/{scan_id}")
async def get_scan_status(scan_id: str):
    return await scanner.get_scan_status(scan_id)


# =====================================================
# Image Protection
# =====================================================
@app.post("/api/protect-image")
async def protect_image(request: ProtectImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=400, detail="Image file not found")

    protected_path = await protector.protect_image(
        request.image_path,
        strength=request.strength,
        method=request.method
    )

    return {
        "status": "success",
        "protected_path": protected_path,
        "project": "SentinelSight"
    }


@app.post("/api/upload-and-protect")
async def upload_and_protect(
    file: UploadFile = File(...),
    method: str = "fawkes",
    strength: Optional[float] = None
):
    file_path = await file_handler.save_upload(file)

    protected_path = await protector.protect_image(
        file_path,
        strength=strength,
        method=method
    )

    return FileResponse(
        protected_path,
        media_type="image/png",
        filename=f"protected_{file.filename}"
    )


# =====================================================
# Batch Analysis & Stats
# =====================================================
@app.post("/api/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    use_external_apis: bool = False
):
    results = []

    for file in files:
        if file_handler.validate_file(file.filename):
            file_path = await file_handler.save_upload(file)
            result = await detector.analyze_image(file_path, use_external_apis)
            results.append(result)
            await file_handler.cleanup_file(file_path)

    return {
        "total_analyzed": len(results),
        "results": results,
        "project": "SentinelSight"
    }


@app.get("/api/stats")
async def get_statistics():
    return await scanner.get_statistics()


# =====================================================
# Directory Browser
# =====================================================
@app.get("/api/browse")
async def browse_directory(path: Optional[str] = None):
    return dir_browser.list_directory(path)


@app.get("/api/common-locations")
async def get_common_locations():
    return {"locations": dir_browser.get_common_locations()}


# =====================================================
# Render Entry Point
# =====================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", settings.API_PORT)),
        workers=1
    )
