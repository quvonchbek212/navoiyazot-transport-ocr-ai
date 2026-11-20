"""OCR uchun yagona vision-klient — Ollama yoki vLLM.

NIMA UCHUN BU MODUL BOR
01.09.2026 gacha to'rtta OCR xizmati (ID-karta, pasport, guvohnoma,
texpasport) modelga TO'RT XIL YO'L bilan borardi: ikkitasi `requests` bilan
`/api/chat` va `/api/generate` ga, ikkitasi esa `ollama` SDK'si orqali.
Server vLLM'ga ko'chganda to'rttasi ham alohida tuzatishni talab qilardi va
biri unutilsa xato faqat ishlab turgan tizimda ko'rinardi.

IKKI PROTOKOL FARQI (bu yerda yashiriladi):

    rasm      Ollama: xabarning `images` ro'yxatida, sof base64
              vLLM:   `content` bo'laklarida, `data:image/...;base64,...` URL

    JSON      Ollama: `format` — sxemaning O'ZI yoki "json"
              vLLM:   `response_format` — OpenAI ko'rinishi

    prefill   Ollama: assistant xabarini oldindan to'ldirish mumkin
                      (thinking'ni o'chirish va JSON'ga majburlash uchun)
              vLLM:   bunday imkon yo'q; o'rniga `response_format` ishlatiladi

Prefill FARQINI CHAQIRUVCHI BILMASLIGI kerak, shuning uchun qaytariladigan
matn ikkala yo'lda ham TO'LIQ bo'ladi: Ollama'da prefill javob boshiga
qo'shib qaytariladi, vLLM'da esa model o'zi to'liq yozadi.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests

from ocr import config


class VisionError(RuntimeError):
    """Model chaqiruvi muvaffaqiyatsiz tugadi."""


def _as_base64(image: str | Path | bytes) -> str:
    """Rasmni base64 ga keltiradi (fayl yo'li, bytes yoki tayyor base64)."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("utf-8")
    # str: fayl yo'li bo'lishi ham, tayyor base64 bo'lishi ham mumkin
    candidate = Path(image)
    try:
        if candidate.is_file():
            return base64.b64encode(candidate.read_bytes()).decode("utf-8")
    except OSError:
        pass
    return image


def _vllm_messages(
    prompt: str, images: Sequence[str], system: str | None
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({
            "type": "image_url",
            # MIME turi jpeg deb beriladi: vLLM tasvirni baribir mazmuni
            # bo'yicha ochadi, sarlavhadagi tur faqat ko'rsatma.
            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
        })
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _ollama_messages(
    prompt: str, images: Sequence[str], system: str | None, prefill: str
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": list(images)})
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def vision_chat(
    prompt: str,
    images: Sequence[str | Path | bytes],
    *,
    system: str | None = None,
    prefill: str = "",
    json_schema: dict[str, Any] | None = None,
    json_mode: bool = False,
    options: dict[str, Any] | None = None,
    timeout: float | None = None,
    seed: int | None = None,
) -> str:
    """Rasm(lar) bilan modelga so'rov yuboradi va javob MATNINI qaytaradi.

    `prefill` faqat Ollama yo'lida ishlatiladi va javob boshiga qo'shib
    qaytariladi — chaqiruvchi uni o'zi qo'shmasligi kerak.
    """
    images_b64 = [_as_base64(i) for i in images]
    options = dict(options or {})
    timeout = timeout or config.TIMEOUT_SECONDS

    if config.USE_VLLM:
        payload: dict[str, Any] = {
            "model": config.VISION_MODEL_NAME,
            "messages": _vllm_messages(prompt, images_b64, system),
            "temperature": options.get("temperature", 0),
            "stream": False,
        }
        if options.get("num_predict"):
            payload["max_tokens"] = int(options["num_predict"])
        if seed is not None:
            payload["seed"] = seed
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "hujjat", "schema": json_schema},
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = f"{config.VLLM_BASE_URL}/chat/completions"
    else:
        payload = {
            "model": config.MODEL_NAME,
            "messages": _ollama_messages(prompt, images_b64, system, prefill),
            "stream": False,
            "think": False,
            "options": {
                "temperature": options.get("temperature", 0),
                "num_ctx": options.get("num_ctx", config.NUM_CTX),
                "num_predict": options.get("num_predict", config.NUM_PREDICT),
                **({"seed": seed} if seed is not None else {}),
            },
        }
        if json_schema:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        url = f"{config.OLLAMA_BASE_URL}/api/chat"

    try:
        response = requests.post(
            url, json=payload, headers=config.llm_headers(), timeout=timeout
        )
    except requests.RequestException as exc:
        raise VisionError(f"LLM serveriga ulanib bo'lmadi ({url}): {exc}") from exc

    if response.status_code != 200:
        snippet = response.text[:200].replace("\n", " ")
        if response.status_code == 404 and config.USE_VLLM:
            raise VisionError(
                f"vLLM'da bunday model yo'q: '{payload.get('model')}'. "
                "Serverdagi nomni ko'ring (GET /v1/models) va .env dagi "
                "VLLM_VISION_MODEL ni to'g'rilang. Nomlarda ikki nuqta emas, "
                "defis ishlatiladi."
            )
        if response.status_code in (401, 403):
            raise VisionError(
                f"LLM kalitni qabul qilmadi ({response.status_code}). "
                ".env dagi VLLM_API_KEY ni tekshiring."
            )
        raise VisionError(f"LLM xatosi {response.status_code}: {snippet}")

    body = response.json()

    if config.USE_VLLM:
        choices = body.get("choices") or []
        if not choices:
            raise VisionError("LLM bo'sh javob qaytardi (choices yo'q).")
        message = choices[0].get("message") or {}
        # Ba'zi modellar butun javobni `reasoning` ga yozadi — yo'qotmaymiz.
        return message.get("content") or message.get("reasoning") or ""

    message = body.get("message") or {}
    # qwen3-vl ba'zan javobni `content` o'rniga `thinking` ga yozadi.
    text = message.get("content") or message.get("thinking") or ""
    return prefill + text
