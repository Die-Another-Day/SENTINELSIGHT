from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # =====================================================
    # API Configuration (Render Compatible)
    # =====================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = int(os.environ.get("PORT", 10000))
    API_WORKERS: int = 1  # Render free tier safe

    # =====================================================
    # Security
    # =====================================================
    SECRET_KEY: str = "change-this-in-production"

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://*.vercel.app",
        "chrome-extension://*"
    ]

    # =====================================================
    # File Handling
    # =====================================================
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"

    # =====================================================
    # Model Paths
    # =====================================================
    MODELS_DIR: str = "models"
    NUDENET_MODEL: str = "models/nudenet_classifier"
    DEEPFAKE_MODEL: str = "models/deepfake_detector"
    CLIP_MODEL: str = "ViT-B/32"

    # =====================================================
    # Detection Thresholds
    # =====================================================
    NSFW_THRESHOLD: float = 0.7
    DEEPFAKE_THRESHOLD: float = 0.75
    DEEPNUDE_THRESHOLD: float = 0.8

    # =====================================================
    # External APIs (Optional)
    # =====================================================
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    AIORNOT_API_KEY: Optional[str] = None

    # =====================================================
    # Image Protection
    # =====================================================
    PERTURBATION_STRENGTH: float = 0.05
    FAWKES_MODE: str = "high"

    # =====================================================
    # Cache (Optional)
    # =====================================================
    USE_CACHE: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # =====================================================
    # Logging
    # =====================================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/sentinelsight.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# =====================================================
# Ensure Required Directories Exist
# =====================================================
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
