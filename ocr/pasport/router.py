"""O'zbekiston pasportini o'qish endpointi."""

from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.concurrency import run_in_threadpool

from ocr.pasport.reader import PassportData, read_passport_bytes
from ocr.uploads import read_image_upload

router = APIRouter(prefix="/pasport", tags=["Pasport"])


class PassportResponse(PassportData):
    """Pasport maydonlari + `debug=true` da modelning xom javobi.

    Xom javob alohida shakl emas, qo'shimcha maydon: aks holda Swagger javob
    sxemasini ko'rsata olmasdi.
    """

    xom_javob: str | None = None


@router.post(
    "/extract",
    response_model=PassportResponse,
    summary="Pasportdan ma'lumotlarni ajratib olish",
    responses={
        413: {"description": "Fayl juda katta"},
        415: {"description": "Rasm formati qo'llab-quvvatlanmaydi"},
        502: {"description": "Ollama serveriga ulanib bo'lmadi"},
        504: {"description": "Model belgilangan vaqtda javob bermadi"},
    },
)
async def extract_passport(
    file: UploadFile = File(..., description="Pasportning ma'lumot sahifasi"),
    debug: bool = Query(
        False, description="True bo'lsa modelning xom javobini ham qaytaradi"),
) -> PassportResponse:
    """Pasportning ma'lumot sahifasini yuklang — maydonlar JSON bo'lib qaytadi.

    MRZ ham o'qiladi; JSHSHIR undan aniqroq olinadi. "Kim tomonidan berilgan"
    maydoni noto'g'ri o'qilsa (masalan markaz nomi yoki tug'ilgan joy tushib
    qolsa) — faqat shu maydon bir marta qayta so'raladi.

    `debug=true` — natija bo'sh chiqqanda model nima qaytarayotganini ko'rish uchun.

    So'rov **1-4 daqiqa** davom etishi mumkin.
    """
    image_bytes = await read_image_upload(file, "file")

    try:
        # read_passport_bytes bloklovchi (requests) — event loopni band qilmaymiz.
        mapped, raw = await run_in_threadpool(read_passport_bytes, image_bytes)
    except requests.Timeout:
        raise HTTPException(504, "Model belgilangan vaqtda javob bermadi")
    except requests.RequestException as e:
        raise HTTPException(502, f"Ollama serveriga ulanib bo'lmadi: {e}")

    return PassportResponse(**mapped, xom_javob=raw if debug else None)
