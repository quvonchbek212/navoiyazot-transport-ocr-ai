"""O'zbekiston texpasporti (avtotransport guvohnomasi) OCR endpointlari.

Bu modul boshqa loyihadan ko'chirilgan edi va `app.security`,
`pasport_ocr.src.*` kabi shu repoda mavjud bo'lmagan paketlarga tayanardi.
Endi u `ocr` paketining umumiy sozlamalari va yuklash tekshiruvidan
foydalanadi — haydovchilik guvohnomasi routeri bilan bir xil.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from ocr import config
from ocr.texpasport.extractor import (
    _ADDR_BOT,
    _ADDR_TOP,
    _RIGHT_ADDR,
    _RIGHT_FRONT,
    TexPasportResponse,
    _save_temp,
    extract_texpasport,
)
from ocr.uploads import read_image_upload

router = APIRouter(prefix="/texpasport", tags=["Texpasport"])

_IMAGE_ACCEPT = ".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"


@router.post(
    "/extract",
    summary="Texpasportdan ma'lumot ajratish",
    response_model=TexPasportResponse,
    responses={
        413: {"description": "Fayl juda katta"},
        415: {"description": "Rasm formati qo'llab-quvvatlanmaydi"},
    },
)
async def extract_texpasport_endpoint(
    front_image: UploadFile = File(
        ...,
        description="Texpasportning OLD tomoni rasmi (majburiy).",
        json_schema_extra={"format": "binary", "accept": _IMAGE_ACCEPT},
    ),
    back_image: UploadFile | None = File(
        default=None,
        description="Texpasportning ORQA tomoni rasmi (ixtiyoriy — 9-19 maydonlar uchun).",
        json_schema_extra={"format": "binary", "accept": _IMAGE_ACCEPT},
    ),
    request_id: str | None = Form(
        default=None,
        description="Ixtiyoriy tashqi so'rov identifikatori.",
    ),
) -> TexPasportResponse:
    """
    Avtomototransport vositasi ro'yxatdan o'tkazilganligi to'g'risidagi guvohnoma
    (texpasport) rasmlarini o'qib **JSON** da qaytaradi.

    | Maydon | Tavsif | Guvohnoma raqami |
    |---|---|---|
    | `plate_number` | Davlat raqam belgisi | 1 |
    | `make_model` | Rusumi / modeli | 2 |
    | `color` | Rangi | 3 |
    | `owner` | Egasi (F.I.O. yoki korxona) | 4 |
    | `address` | Manzili | 5 |
    | `date_of_issue` | Berilgan sanasi DD.MM.YYYY | 6 |
    | `issuing_authority` | YHX boshqarmasi bo'limi | 7 |
    | `pinfl_tin` | JSHSHIR / STIR | 8 |
    | `year_of_manufacture` | Ishlab chiqarilgan yili | 9 (orqa) |
    | `vehicle_type` | Turi | 10 (orqa) |
    | `vin` | Kuzov / shassi raqami | 11 (orqa) |
    | `max_mass_kg` | To'la vazni (kg) | 12 (orqa) |
    | `unladen_mass_kg` | Yuksiz vazni (kg) | 13 (orqa) |
    | `engine_number` | Dvigatel raqami | 14 (orqa) |
    | `engine_capacity` | Dvigatel quvvati | 15 (orqa) |
    | `fuel_type` | Yonilg'i turi | 16 (orqa) |
    | `seats` | O'tiradigan joylar soni | 17 (orqa) |
    | `standing_places` | Tik turadigan joylar soni | 18 (orqa) |
    | `special_marks` | Alohida belgilar | 19 (orqa) |
    """
    front_bytes = await read_image_upload(front_image, "front_image")
    back_bytes = (
        await read_image_upload(back_image, "back_image") if back_image is not None else None
    )

    data, warnings = await run_in_threadpool(
        extract_texpasport,
        front_bytes,
        back_bytes,
        ollama_host=config.OLLAMA_BASE_URL,
        ollama_model=config.MODEL_NAME,
        timeout_seconds=config.TIMEOUT_SECONDS,
        num_ctx=config.NUM_CTX,
        num_predict=config.NUM_PREDICT,
        ngrok_skip=config.NGROK_SKIP_BROWSER_WARNING,
    )

    return TexPasportResponse(
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
    front_image: UploadFile = File(..., description="Texpasportning OLD tomoni"),
    crop: str = Form(default="addr", description="front | addr"),
) -> Response:
    """
    Krop chegaralarini tekshirish uchun: cropning JPEG rasmini qaytaradi.
    Swagger dan foydalanib, qaysi maydon ko'rinayotganini tekshiring.
    """
    front_bytes = await read_image_upload(front_image, "front_image")

    crop_map = {
        "front": (0.0, 1.0, _RIGHT_FRONT),
        "addr": (_ADDR_TOP, _ADDR_BOT, _RIGHT_ADDR),
    }
    top, bot, right = crop_map.get(crop, crop_map["addr"])

    path = _save_temp(front_bytes, top_pct=top, bottom_pct=bot, right_pct=right)
    try:
        img_bytes = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)

    return Response(content=img_bytes, media_type="image/jpeg")
