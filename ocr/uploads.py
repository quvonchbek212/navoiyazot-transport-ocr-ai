"""Yuklangan rasm fayllarini tekshirish — uch OCR routeri uchun umumiy.

Tekshiruv model chaqirilishidan OLDIN bo'lishi muhim: bitta OCR so'rovi
1-4 daqiqa oladi, shuning uchun noto'g'ri fayl uchun shuncha kutish va
GPU vaqtini sarflash mantiqsiz.
"""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

from ocr.config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_BYTES


async def read_image_upload(file: UploadFile, label: str) -> bytes:
    """Rasmni o'qiydi; tur/hajm mos kelmasa HTTPException ko'taradi."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            415,
            f"{label}: rasm formati qo'llab-quvvatlanmaydi ({file.content_type}). "
            f"Ruxsat: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}",
        )
    try:
        data = await file.read()
    finally:
        await file.close()

    if not data:
        raise HTTPException(400, f"{label}: fayl bo'sh")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"{label}: fayl {MAX_UPLOAD_BYTES // 1024 // 1024} MB dan katta"
        )
    return data
