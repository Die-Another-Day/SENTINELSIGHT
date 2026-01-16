import base64
import httpx
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

# Optional AI providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from config import settings


class AIAPIsIntegration:
    """
    SentinelSight – External AI API Integrations
    Enhances internal detection using trusted third-party vision models
    """

    def __init__(self):
        self.openai_available = False
        self.google_available = False
        self.anthropic_available = False
        self.anthropic_client = None

        self._init_openai()
        self._init_google()
        self._init_anthropic()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_openai(self):
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_available = True
            logger.info("SentinelSight: OpenAI Vision enabled")

    def _init_google(self):
        if GOOGLE_AVAILABLE and settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.google_available = True
            logger.info("SentinelSight: Google Gemini enabled")

    def _init_anthropic(self):
        if ANTHROPIC_AVAILABLE and settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(
                api_key=settings.ANTHROPIC_API_KEY
            )
            self.anthropic_available = True
            logger.info("SentinelSight: Anthropic Claude enabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_image(self, image_path: str) -> Dict:
        """
        Run image analysis using all available external AI services
        """

        results, scores, details = {}, {}, {}

        if self.openai_available:
            res = await self._analyze_with_openai(image_path)
            if res:
                results["openai"] = res
                scores["openai"] = res["score"]
                details["openai"] = res["details"]

        if self.google_available:
            res = await self._analyze_with_gemini(image_path)
            if res:
                results["gemini"] = res
                scores["gemini"] = res["score"]
                details["gemini"] = res["details"]

        if self.anthropic_available:
            res = await self._analyze_with_claude(image_path)
            if res:
                results["claude"] = res
                scores["claude"] = res["score"]
                details["claude"] = res["details"]

        if settings.AIORNOT_API_KEY:
            res = await self._analyze_with_aiornot(image_path)
            if res:
                results["aiornot"] = res
                scores["aiornot"] = res["score"]
                details["aiornot"] = res["details"]

        return {
            "scores": scores,
            "details": details,
            "raw_results": results
        }

    # ------------------------------------------------------------------
    # Provider Implementations
    # ------------------------------------------------------------------

    async def _analyze_with_openai(self, image_path: str) -> Optional[Dict]:
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image for NSFW content, AI generation, "
                                "deepfakes, violence, or disturbing imagery. "
                                "Respond clearly with reasoning."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }],
                max_tokens=400
            )

            text = response.choices[0].message.content.lower()
            score = 0.5 if any(w in text for w in ["nsfw", "fake", "violent", "inappropriate"]) else 0.0

            return {"score": score, "details": {"response": text}}

        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return None

    async def _analyze_with_gemini(self, image_path: str) -> Optional[Dict]:
        try:
            from PIL import Image

            model = genai.GenerativeModel("gemini-pro-vision")
            img = Image.open(image_path)

            response = model.generate_content([
                "Analyze this image for NSFW, deepfake, or violent content.",
                img
            ])

            text = response.text.lower()
            score = 0.5 if any(w in text for w in ["nsfw", "fake", "violent"]) else 0.0

            return {"score": score, "details": {"response": response.text}}

        except Exception as e:
            logger.error(f"Gemini Vision error: {e}")
            return None

    async def _analyze_with_claude(self, image_path: str) -> Optional[Dict]:
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp"
            }.get(Path(image_path).suffix.lower(), "image/jpeg")

            msg = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Check for NSFW, deepfake, violence or disturbing content."
                            )
                        }
                    ]
                }]
            )

            text = msg.content[0].text.lower()
            score = 0.5 if any(w in text for w in ["nsfw", "fake", "violent"]) else 0.0

            return {"score": score, "details": {"response": msg.content[0].text}}

        except Exception as e:
            logger.error(f"Claude Vision error: {e}")
            return None

    async def _analyze_with_aiornot(self, image_path: str) -> Optional[Dict]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                with open(image_path, "rb") as f:
                    response = await client.post(
                        "https://api.aiornot.com/v1/reports/image",
                        files={"image": f},
                        headers={
                            "Authorization": f"Bearer {settings.AIORNOT_API_KEY}"
                        }
                    )

            if response.status_code == 200:
                data = response.json()
                return {
                    "score": data.get("ai_probability", 0.0),
                    "details": data
                }

        except Exception as e:
            logger.error(f"AIorNot API error: {e}")

        return None
