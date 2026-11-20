# -*- coding: utf-8 -*-
"""
O'zbekiston ko'k pasportining ma'lumot sahifasini Ollama vision modeli bilan
o'qish. HTTP qatlami bu yerda emas — u `ocr/pasport/router.py` da.

CLI sifatida (loyiha ildizidan):
    python -m ocr.pasport.reader pasport.jpg
    python -m ocr.pasport.reader pasport.jpg --out natija.json
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import requests
from pydantic import BaseModel
from typing import Optional

from ocr import config
from ocr.llm import VisionError, vision_chat

# ----------------------------------------------------------------------------
# Sozlamalar
# ----------------------------------------------------------------------------
OLLAMA_URL = config.OLLAMA_BASE_URL
MODEL = config.MODEL_NAME
TIMEOUT = config.TIMEOUT_SECONDS
HEADERS = config.ollama_headers()
MAX_SIDE = 2400  # mayda yozilgan beruvchi organ qiymatini saqlash uchun

PROMPT = """This image is the data page of a passport of the Republic of Uzbekistan.
Read the visible text carefully and extract all fields listed below. Also read the
two-line MRZ at the bottom of the passport (the machine-readable zone containing
the << characters).

CRITICAL INSTRUCTION FOR "kim_tomonidan_berilgan": locate the exact printed label
"KIM TOMONIDAN BERILGAN" in the image. Extract only the value printed directly
below that label. Copy the complete value exactly as printed, without shortening,
translating, correcting, or guessing it. The value is normally written in uppercase
and may contain a region, district or city name and the abbreviation IIB.
Do not return the label text "KIM TOMONIDAN BERILGAN" as the value.
Do not use "STATE PERSONALIZATION CENTRE", the place of birth, or any nearby field
as this value. If the exact label or its value is not clearly visible, return null.

Return ONLY a JSON object in the following form:
{
  "hujjat_turi": "Document type (for example P)",
  "davlat_kodi": "Country code (for example UZB)",
  "pasport_raqami": "Passport No",
  "familiya": "Surname",
  "ism": "Given names",
  "millati": "Nationality",
  "jinsi": "Sex (M or F)",
  "tugilgan_sana": "Date of birth DD.MM.YYYY",
  "tugilgan_joyi": "Place of birth",
  "jshshir_pinfl": "Personal No / JShShIR (14 digits)",
  "berilgan_sana": "Date of issue DD.MM.YYYY",
  "amal_qilish_muddati": "Date of expiry DD.MM.YYYY",
  "kim_tomonidan_berilgan": "Complete value printed directly below the KIM TOMONIDAN BERILGAN label",
  "mrz": "Both MRZ lines"
}
Use null for any field that cannot be read. Do not output explanations, Markdown,
or any text outside the JSON object."""


# Model boshqa nom bilan qaytarishi mumkin — shu jadval orqali moslаymiz
KEY_ALIASES = {
    "hujjat_turi": ["type", "document_type", "doc_type", "hujjat_turi", "turi"],
    "davlat_kodi": ["country_code", "country", "issuing_country", "davlat_kodi", "davlat"],
    "pasport_raqami": ["passport_no", "passport_number", "document_no", "number", "pasport_raqami", "raqam"],
    "familiya": ["surname", "last_name", "familiya", "familiyasi"],
    "ism": ["given_names", "given_name", "first_name", "name", "ism", "ismi"],
    "millati": ["nationality", "millati", "millati"],
    "jinsi": ["sex", "gender", "jinsi"],
    "tugilgan_sana": ["date_of_birth", "birth_date", "dob", "tugilgan_sana"],
    "tugilgan_joyi": ["place_of_birth", "birth_place", "pob", "tugilgan_joyi"],
    "jshshir_pinfl": ["personal_no", "personal_number", "pinfl", "jshshir", "jshshir_pinfl", "id_number"],
    "berilgan_sana": ["date_of_issue", "issue_date", "berilgan_sana"],
    "amal_qilish_muddati": ["date_of_expiry", "expiry_date", "expiration_date", "amal_qilish_muddati"],
    "kim_tomonidan_berilgan": ["kim_tomonidan_berilgan"],
    "mrz": ["mrz", "machine_readable_zone", "mrz_lines"],
}


# ----------------------------------------------------------------------------
# Javob modeli
# ----------------------------------------------------------------------------
class PassportData(BaseModel):
    hujjat_turi: Optional[str] = None
    davlat_kodi: Optional[str] = None
    pasport_raqami: Optional[str] = None
    familiya: Optional[str] = None
    ism: Optional[str] = None
    millati: Optional[str] = None
    jinsi: Optional[str] = None
    tugilgan_sana: Optional[str] = None
    tugilgan_joyi: Optional[str] = None
    jshshir_pinfl: Optional[str] = None
    berilgan_sana: Optional[str] = None
    amal_qilish_muddati: Optional[str] = None
    kim_tomonidan_berilgan: Optional[str] = None
    mrz: Optional[str] = None


# ----------------------------------------------------------------------------
# Yordamchi funksiyalar
# ----------------------------------------------------------------------------
def optimize_image(image_bytes: bytes) -> bytes:
    """Rasmni kichraytiradi va JPEG qiladi. Pillow bo'lmasa asl holicha qaytaradi."""
    try:
        from PIL import Image
    except ImportError:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return image_bytes


def extract_json(text: str) -> dict:
    """Model javobidan JSON qismini ajratib oladi."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_keys(data: dict) -> dict:
    """Model qaytargan kalitlarni bizning standart kalitlarga moslashtiradi."""
    # kichik harf + probel/tirelarni pastki chiziqqa aylantirib qidiramiz
    lowered = {re.sub(r"[\s\-]+", "_", str(k).strip().lower()): v for k, v in data.items()}
    result = {}
    for our_key, aliases in KEY_ALIASES.items():
        for alias in aliases:
            normalized_alias = re.sub(r"[\s\-]+", "_", alias.strip().lower())
            if normalized_alias in lowered and lowered[normalized_alias] not in (None, "", "null"):
                val = lowered[normalized_alias]
                result[our_key] = " ".join(val) if isinstance(val, list) else str(val)
                break
    return result


def _comparable_text(value) -> str:
    """Ikki OCR qiymatini tinish belgilari va registrsiz solishtirish uchun tayyorlaydi."""
    # \w Unicode harflarini ham saqlaydi; aks holda kirill yozuvidagi IIB nomi
    # bo'sh qiymatga aylanib, xato ravishda rad etilgan bo'lardi.
    return re.sub(r"[\W_]+", "", str(value or "").casefold(), flags=re.UNICODE)


def clean_issuing_authority(authority) -> Optional[str]:
    """Model qiymatga qo'shib yuborgan maydon yorlig'ini olib tashlaydi."""
    if authority is None:
        return None
    value = str(authority).strip()
    value = re.sub(
        r"^\s*KIM[\s_\-]+TOMONIDAN[\s_\-]+BERILGAN\s*[:\-–—]*\s*",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return value or None


def invalid_issuing_authority(authority, birth_place=None) -> bool:
    """Beruvchi organ o'rniga olinib qoladigan ma'lum noto'g'ri qiymatlarni aniqlaydi."""
    authority_text = _comparable_text(authority)
    if not authority_text:
        return True

    # Bu umumiy markaz nomi API uchun kerakli hududiy organ/IIB emas.
    blocked_centres = (
        "statepersonalizationcentre",
        "statepersonalizationcenter",
    )
    if any(blocked in authority_text for blocked in blocked_centres):
        return True

    birth_place_text = _comparable_text(birth_place)
    if birth_place_text and authority_text == birth_place_text:
        return True

    return False


AUTHORITY_RETRY_PROMPT = """Inspect this Uzbekistan passport image and locate the
exact printed label "KIM TOMONIDAN BERILGAN". Extract only the value printed
directly below that label. Copy the complete value exactly as printed, without
shortening, translating, correcting, or guessing it. The value is normally in
uppercase and may contain a region, district or city name and the abbreviation IIB.

"KIM TOMONIDAN BERILGAN" is only the field label. Never return the label itself
as the value. Do not use "STATE PERSONALIZATION CENTRE", the place of birth, or
any nearby field. If the exact label or its value is not clearly visible, use null.

Return only a JSON object containing exactly one key named
"kim_tomonidan_berilgan". Do not output any explanation or Markdown.
"""


def pinfl_from_mrz(mrz) -> Optional[str]:
    """MRZ raqamlaridan JShShIR ni ajratadi: oxiridan 2 raqamni tashlab,
    undan oldingi 14 raqamni oladi (TD3: oxirgi 2 — nazorat, undan oldingi 14 — shaxsiy raqam)."""
    if not mrz:
        return None
    digits = re.sub(r"\D", "", str(mrz))  # faqat raqamlarni qoldiramiz
    if len(digits) < 16:
        return None
    return digits[-16:-2]


def read_passport_bytes(image_bytes: bytes):
    """Rasmni Ollama vision modeliga yuboradi. (mapped_dict, raw_response) qaytaradi."""
    image_bytes = optimize_image(image_bytes)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    raw = vision_chat(
        PROMPT,
        [img_b64],
        json_mode=True,
        options={"temperature": 0, "num_ctx": 4096},
        timeout=TIMEOUT,
    )
    parsed = extract_json(raw)
    mapped = normalize_keys(parsed)
    cleaned_authority = clean_issuing_authority(
        mapped.get("kim_tomonidan_berilgan")
    )
    if cleaned_authority:
        mapped["kim_tomonidan_berilgan"] = cleaned_authority
    else:
        mapped.pop("kim_tomonidan_berilgan", None)

    # Model umumiy markaz yoki tug'ilgan joyni beruvchi organ deb olgan bo'lsa,
    # aynan shu maydonni bir marta yo'naltirilgan prompt bilan qayta o'qiymiz.
    if invalid_issuing_authority(
        mapped.get("kim_tomonidan_berilgan"), mapped.get("tugilgan_joyi")
    ):
        mapped.pop("kim_tomonidan_berilgan", None)
        retry_raw = vision_chat(
            AUTHORITY_RETRY_PROMPT,
            [img_b64],
            json_mode=True,
            options={"temperature": 0, "num_ctx": 2048},
            timeout=TIMEOUT,
        )
        retry_mapped = normalize_keys(extract_json(retry_raw))
        retry_authority = clean_issuing_authority(
            retry_mapped.get("kim_tomonidan_berilgan")
        )
        if not invalid_issuing_authority(retry_authority, mapped.get("tugilgan_joyi")):
            mapped["kim_tomonidan_berilgan"] = retry_authority
        raw = f"{raw}\n\n[authority_retry]\n{retry_raw}"

    # JShShIR ni MRZ dan aniqroq olamiz (oxiridan 2 tashlab, 14 raqam)
    pinfl = pinfl_from_mrz(mapped.get("mrz"))
    if pinfl:
        mapped["jshshir_pinfl"] = pinfl
    return mapped, raw


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="O'zbekiston pasportidan ma'lumot o'qish (qwen3-vl)"
    )
    parser.add_argument("image", type=Path, help="Pasport rasm fayli (jpg/png)")
    parser.add_argument("--out", type=Path, help="Natijani saqlash uchun JSON fayl")
    parser.add_argument("--raw", action="store_true", help="Modelning xom javobini ham chiqarish")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Xato: '{args.image}' fayli topilmadi.", file=sys.stderr)
        return 1

    print(f"[*] Rasm yuborilmoqda: {args.image.name} ...", file=sys.stderr)
    try:
        mapped, raw = read_passport_bytes(args.image.read_bytes())
    except (VisionError, requests.RequestException) as e:
        # `vision_chat` ulanish xatosini `VisionError` ga o'raydi — eski
        # `requests.RequestException` bu yerga endi yetib kelmaydi.
        print(f"Server xatosi: {e}", file=sys.stderr)
        return 1

    output = json.dumps(PassportData(**mapped).model_dump(), ensure_ascii=False, indent=2)
    print(output)
    if args.raw:
        print(f"\n[xom javob]\n{raw}", file=sys.stderr)

    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"\n[*] Natija saqlandi: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
