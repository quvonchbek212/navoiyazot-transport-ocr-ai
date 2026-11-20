"""O'zbekiston ID-kartasini o'qish endpointi.

Router `main.py` dagi yagona FastAPI ilovasiga ulanadi — bu yerda alohida
`FastAPI()` yaratilmaydi, aks holda Swagger bir nechta bo'lib ketardi.
"""

from __future__ import annotations

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ocr.id_card import extractor
from ocr.uploads import read_image_upload

router = APIRouter(prefix="/id-card", tags=["ID-karta"])


class IdCardFields(BaseModel):
    """Swagger javob sxemasi shu modeldan chiziladi."""

    document_number: str | None = Field(
        None, description="Karta raqami", examples=["AD8718879"])
    pinfl: str | None = Field(
        None, description="JSHSHIR, 14 raqam", examples=["40305986080012"])
    issue_date: str | None = Field(
        None, description="Berilgan sanasi", examples=["2024-09-21"])
    expiry_date: str | None = Field(
        None, description="Amal qilish muddati", examples=["2034-09-20"])
    issued_by: str | None = Field(
        None, description="Kim tomonidan berilgan", examples=["IIV 18224"])


@router.post(
    "/extract",
    response_model=IdCardFields,
    summary="ID-kartadan ma'lumotlarni ajratib olish",
    responses={
        413: {"description": "Fayl juda katta"},
        415: {"description": "Rasm formati qo'llab-quvvatlanmaydi"},
        502: {"description": "Ollama serveriga ulanib bo'lmadi"},
        504: {"description": "Model belgilangan vaqtda javob bermadi"},
    },
)
async def extract_id_card(
    front: UploadFile = File(..., description="Kartaning old tomoni"),
    back: UploadFile | None = File(
        None, description="Orqa tomoni — MRZ shu yerda, tavsiya etiladi"),
) -> IdCardFields:
    """Kartaning old (va ixtiyoriy ravishda orqa) tomonini yuboring.

    Orqa tomon berilsa MRZ ham o'qiladi va checksumdan o'tsa qiymatlar
    jimgina tuzatiladi. O'qilmagan maydon `null` bo'lib qaytadi.

    So'rov **1-4 daqiqa** davom etishi mumkin.

    **Muhim:** natija OCR modelining o'qishi — u xato bo'lishi mumkin va
    javobning o'zidan buni bilib bo'lmaydi. Yuridik ahamiyatga ega ishlarda
    odam tekshirishi shart.
    """
    front_bytes = await read_image_upload(front, "front")
    back_bytes = await read_image_upload(back, "back") if back is not None else None

    try:
        # extractor.extract bloklovchi (requests) — event loopni band qilmaymiz.
        fields = await run_in_threadpool(extractor.extract, front_bytes, back_bytes)
    except requests.Timeout:
        raise HTTPException(504, "Model belgilangan vaqtda javob bermadi")
    except requests.RequestException as e:
        raise HTTPException(502, f"Ollama serveriga ulanib bo'lmadi: {e}")
    except (RuntimeError, ValueError) as e:
        raise HTTPException(502, f"Model javobini o'qib bo'lmadi: {e}")

    return IdCardFields(**fields)
