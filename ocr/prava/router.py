"""O'zbekiston haydovchilik guvohnomasi OCR endpointlari.

Bu modul boshqa loyihadan ko'chirilgan edi va `app.security`,
`app.services.*`, `pasport_ocr.src.*` kabi shu repoda mavjud bo'lmagan
paketlarga tayanardi. Endi u `ocr` paketining umumiy sozlamalari va yuklash
tekshiruvidan foydalanadi.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from ocr import config
from ocr.prava.extractor import (
    DrivingLicenseResponse,
    _F3_BOT, _F3_TOP, _F4_BOT, _F4_TOP, _F8_BOT, _F8_TOP,
    _RIGHT_F3F4, _RIGHT_F8,
    _save_temp,
    extract_driving_license,
)
from ocr.uploads import read_image_upload

router = APIRouter(prefix="/driving-license", tags=["Haydovchilik guvohnomasi"])

_IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"


@router.post(
    "/extract",
    summary="Haydovchilik guvohnomasidan ma'lumot ajratish",
    response_model=DrivingLicenseResponse,
    responses={
        413: {"description": "Fayl juda katta"},
        415: {"description": "Rasm formati qo'llab-quvvatlanmaydi"},
    },
)
async def extract_driving_license_endpoint(
    front_image: UploadFile = File(
        ...,
        description="Guvohnomaning OLD tomoni rasmi (majburiy).",
        json_schema_extra={"format": "binary", "accept": _IMAGE_ACCEPT},
    ),
    back_image: UploadFile | None = File(
        default=None,
        description="Guvohnomaning ORQA tomoni rasmi (ixtiyoriy — toifalar jadvali uchun).",
        json_schema_extra={"format": "binary", "accept": _IMAGE_ACCEPT},
    ),
    request_id: str | None = Form(
        default=None,
        description="Ixtiyoriy tashqi so'rov identifikatori.",
    ),
) -> DrivingLicenseResponse:
    """
    Uzbekiston haydovchilik guvohnomasi rasmlarini o'qib **JSON** da qaytaradi.

    | Maydon | Tavsif | Guvohnoma raqami |
    |---|---|---|
    | `surname` | Familiya | 1 — FAMILIYASI |
    | `given_names` | Ismi va otasining ismi | 2 |
    | `place_of_birth` | Tug'ilgan joyi | 3 |
    | `date_of_birth` | Tug'ilgan sanasi DD.MM.YYYY | 3 |
    | `date_of_issue` | Berilgan sanasi DD.MM.YYYY | 4a |
    | `date_of_expiry` | Amal qilish muddati DD.MM.YYYY | 4b |
    | `place_of_issue` | Berilgan joyi | 4c |
    | `license_number` | Guvohnoma raqami | 5 |
    | `residence` | Yashash joyi (to'liq manzil) | 8 |
    | `categories` | Toifalar ro'yxati | 9 |
    | `category_details` | Har bir toifa uchun sanalar | Orqa tomon |
    """
    front_bytes = await read_image_upload(front_image, "front_image")
    back_bytes = (
        await read_image_upload(back_image, "back_image") if back_image is not None else None
    )

    data, warnings = await run_in_threadpool(
        extract_driving_license,
        front_bytes,
        back_bytes,
        ollama_host=config.OLLAMA_BASE_URL,
        ollama_model=config.MODEL_NAME,
        timeout_seconds=config.TIMEOUT_SECONDS,
        num_ctx=config.NUM_CTX,
        num_predict=config.NUM_PREDICT,
        ngrok_skip=config.NGROK_SKIP_BROWSER_WARNING,
    )

    return DrivingLicenseResponse(
        status="partial" if warnings else "success",
        model=config.MODEL_NAME,
        data=data,
        warnings=warnings,
        request_id=request_id,
    )


@router.post(
    "/debug-crops",
    summary="[DEBUG] Krop rasmlarini ko'rish — chegaralarni tekshirish uchun",
    response_class=Response,
)
async def debug_crops(
    front_image: UploadFile = File(..., description="Guvohnomaning OLD tomoni"),
    crop: str = Form(default="f3", description="f3 | f4 | f8"),
) -> Response:
    """
    Krop chegaralarini tekshirish uchun: cropning JPEG rasmini qaytaradi.
    Swagger dan foydalanib, qaysi maydon ko'rinayotganini tekshiring.
    """
    front_bytes = await read_image_upload(front_image, "front_image")

    crop_map = {
        "f3": (_F3_TOP, _F3_BOT, _RIGHT_F3F4),
        "f4": (_F4_TOP, _F4_BOT, _RIGHT_F3F4),
        "f8": (_F8_TOP, _F8_BOT, _RIGHT_F8),
    }
    top, bot, right = crop_map.get(crop, crop_map["f3"])

    path = _save_temp(front_bytes, top_pct=top, bottom_pct=bot, right_pct=right)
    try:
        img_bytes = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)

    return Response(content=img_bytes, media_type="image/jpeg")
