"""OCR xizmatlarining umumiy sozlamalari — loyiha ildizidagi `.env` dan.

Uch xizmat (ID-karta, pasport, guvohnoma) ilgari alohida loyiha bo'lgani uchun
har biri Ollama manzilini o'z faylida qattiq yozib qo'ygan edi. Manzil ngrok
tunneli bo'lgani uchun u tez-tez o'zgaradi va uch joyni qidirib chiqishga to'g'ri
kelardi. Endi manba bitta: `.env` dagi `OLLAMA_BASE_URL`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------- Ollama
# Chatbot bilan bitta server. Model esa boshqa: chatbotga SQL yozish kerak,
# OCR ga rasm o'qish — shuning uchun CHATBOT_OLLAMA_MODEL emas.
OLLAMA_BASE_URL = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

MODEL_NAME = os.getenv("OCR_OLLAMA_MODEL") or "qwen3-vl:8b"

# ----------------------------------------------------------------------- vLLM
# Server 01.09.2026 da vLLM'ga ko'chdi. To'ldirilgan bo'lsa OCR ham shu yerga
# boradi va yuqoridagi OLLAMA_* manzili ishlatilmaydi.
#
# DIQQAT — MODEL ALMASHADI: Ollama'da OCR alohida vision model ishlatardi
# (qwen3-vl:8b), vLLM'da esa serverda BITTA model turibdi va u matn uchun ham,
# rasm uchun ham o'sha — gemma4-31b. U multimodal ekani 01.09.2026 da
# tekshirilgan (image_url bilan rasmni tavsiflab berdi), ammo bu BOSHQA model:
# hujjat o'qish sifati qwen3-vl bilan bir xil bo'lishi shart emas.
# Serverda alohida vision model paydo bo'lsa: VLLM_VISION_MODEL=<nomi>
VLLM_BASE_URL = (os.getenv("VLLM_BASE_URL") or "").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY") or ""
USE_VLLM = bool(VLLM_BASE_URL)

VISION_MODEL_NAME = (
    os.getenv("VLLM_VISION_MODEL") or os.getenv("VLLM_MODEL") or MODEL_NAME
)

# ngrok bepul tunneli so'rovni aynan 300 s da uzadi, shuning uchun undan pastroq.
TIMEOUT_SECONDS = int(os.getenv("OCR_OLLAMA_TIMEOUT_SECONDS") or 280)
NUM_CTX = int(os.getenv("OCR_OLLAMA_NUM_CTX") or 8192)
NUM_PREDICT = int(os.getenv("OCR_OLLAMA_NUM_PREDICT") or 1024)

# ngrok bepul tunneli birinchi so'rovga HTML ogohlantirish sahifasini qaytaradi;
# bu sarlavha usiz JSON o'rniga HTML kelishiga sabab bo'ladi.
NGROK_SKIP_BROWSER_WARNING = _flag("OLLAMA_NGROK_SKIP_BROWSER_WARNING", True)

# ID-karta MRZ zonasini o'qish ikkinchi model chaqiruvini talab qiladi — javob
# taxminan ikki barobar sekinlashadi. Evaziga MRZ checksumdan o'tsa qiymatlar
# tuzatiladi. Tezlik muhimroq bo'lsa: OCR_USE_MRZ=false
USE_MRZ = _flag("OCR_USE_MRZ", True)


def llm_headers() -> dict[str, str]:
    """LLM so'rovlariga qo'shiladigan sarlavhalar (vLLM'da API kalit ham)."""
    base = VLLM_BASE_URL if USE_VLLM else OLLAMA_BASE_URL
    headers: dict[str, str] = {}
    if NGROK_SKIP_BROWSER_WARNING and "ngrok" in base.lower():
        headers["ngrok-skip-browser-warning"] = "true"
    if USE_VLLM and VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


def ollama_headers() -> dict[str, str]:
    """Eski nom — ko'chirilmagan chaqiruvlar uchun saqlangan."""
    return llm_headers()


# --------------------------------------------------------------- Yuklanadigan
MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_MB") or 10) * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
