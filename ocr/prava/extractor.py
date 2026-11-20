"""Uzbekiston haydovchilik guvohnomasi — Ollama vision orqali ma'lumot ajratish."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ocr.llm import vision_chat


# ---------------------------------------------------------------------------
# Sxemalar
# ---------------------------------------------------------------------------

class CategoryDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str | None = None
    date_of_issue: str | None = None
    date_of_expiry: str | None = None
    additional_info: str | None = None


class DrivingLicenseData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    surname: str | None = Field(None, description="Familiya (1 — FAMILIYASI)")
    given_names: str | None = Field(None, description="Ismi va otasining ismi (2)")
    place_of_birth: str | None = Field(None, description="Tug'ilgan joyi (3)")
    date_of_birth: str | None = Field(None, description="Tug'ilgan sanasi DD.MM.YYYY (3)")
    date_of_issue: str | None = Field(None, description="Berilgan sanasi DD.MM.YYYY (4a)")
    date_of_expiry: str | None = Field(None, description="Amal qilish muddati DD.MM.YYYY (4b)")
    place_of_issue: str | None = Field(None, description="Berilgan joyi (4c)")
    pinfl: str | None = Field(None, description="PINFL — shaxsiy identifikatsiya raqami (4d)")
    license_number: str | None = Field(None, description="Guvohnoma raqami (5)")
    residence: str | None = Field(None, description="Yashash joyi to'liq manzil (8)")
    categories: list[str] | None = Field(None, description="Toifalar ro'yxati (9)")
    category_details: list[CategoryDetail] = Field(
        default_factory=list,
        description="Har bir toifa uchun sanalar (orqa tomon jadvali)",
    )


class DrivingLicenseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = Field("success", description="'success' yoki 'partial'")
    model: str = ""
    data: DrivingLicenseData = Field(default_factory=DrivingLicenseData)
    warnings: list[str] = Field(default_factory=list)
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Promptlar
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise OCR assistant for Uzbekistan government documents. "
    "Read text exactly as printed — do not guess or paraphrase. "
    "Return ONLY valid JSON. No markdown, no explanation."
)

# DIQQAT: har bir prompt OXIRIDA kerakli JSON kalitlari aniq sanab o'tilishi
# shart. Ilgari kalit nomlarini Ollama'ning `format` sxemasi majburlar edi;
# u olib tashlangach (sabab `_NO_THINK` izohida) model kalitlarni o'zicha
# tanlay boshladi — maydon raqamlari bilan: {"1": "TESTOV", "4a": "12.09.2020"}.
# Natijada ma'lumot to'g'ri o'qilsa ham, hech bir maydon joyiga tushmaydi.
#
# Promptlarning uzunligi endi muhim emas: prefill o'ylashni o'chirgani uchun
# model ko'rsatmalar ustida "mulohaza yuritmaydi" — ID-kartaning 30 qatorlik
# prompti ham ~2 s da ishlaydi.

_FRONT_PROMPT = (
    "This is the FRONT side of an Uzbekistan driving licence (Haydovchilik Guvohnomasi).\n\n"
    "Find each numbered field label and copy the value printed next to or below it:\n\n"
    "  surname        ← label '1.'\n"
    "  given_names    ← label '2.' — may be two lines, join with space\n"
    "  place_of_birth ← label '3.' — the place name only (not the date)\n"
    "  date_of_birth  ← label '3.' — the date in DD.MM.YYYY\n"
    "  date_of_issue  ← label '4a.' — DD.MM.YYYY\n"
    "  date_of_expiry ← label '4b.' — DD.MM.YYYY\n"
    "  place_of_issue ← label '4c.'\n"
    "  pinfl          ← label '4d.' — digits only, no spaces\n"
    "  license_number ← label '5.'\n"
    "  residence      ← label '8.' — all lines joined with ', '\n"
    "  categories     ← label '9.' — list of codes e.g. [\"B\"]\n\n"
    "- Copy text EXACTLY as printed.\n"
    "- Set null if a field is unreadable.\n"
    "- category_details must be [].\n\n"
    "Return a JSON object with exactly these keys: surname, given_names, "
    "place_of_birth, date_of_birth, date_of_issue, date_of_expiry, place_of_issue, "
    "pinfl, license_number, residence, categories, category_details."
)

# Field 3 uchun alohida krop (faqat joy nomi)
_F3_PROMPT = (
    "This image shows a section of an Uzbekistan driving licence.\n"
    "Find the field labelled '3.' (the number three followed by a dot or period).\n\n"
    "Field '3.' contains: a PLACE NAME followed by a DATE on the same line.\n"
    "Example format: '3. TOSHKENT 01.01.1990'\n\n"
    "Copy ONLY the place name — the word(s) that appear BEFORE the date.\n"
    "Do NOT copy the date. Do NOT copy text from field '1.' or '2.' above.\n"
    "The place name is typically one or two words (a city or district name).\n\n"
    "Return a JSON object with exactly this key: place_of_birth"
)

# Fields 4a, 4b, 4c, 4d uchun alohida krop
_F4_PROMPT = (
    "This image shows a section of an Uzbekistan driving licence.\n"
    "Find ONLY the fields labelled '4a.', '4b.', '4c.', '4d.' and copy their values.\n\n"
    "  '4a.' → date_of_issue — the date AFTER label '4a.', format DD.MM.YYYY.\n"
    "          WARNING: field '3.' above also has a date — ignore that one.\n"
    "  '4b.' → date_of_expiry — DD.MM.YYYY, year is typically 10 larger than 4a.\n"
    "  '4c.' → place_of_issue — the COMPLETE text, every word and abbreviation\n"
    "          (DXM, IIB, TUMANI, SHAHRI). Do NOT stop mid-text.\n"
    "  '4d.' → pinfl — ALL digits after the label, left to right, no spaces.\n\n"
    "Set null if a field is unreadable.\n"
    "Return a JSON object with exactly these keys: date_of_issue, date_of_expiry, "
    "place_of_issue, pinfl."
)

# Field 8 uchun alohida krop (o'ng tomon ro'yxatsiz)
_F8_PROMPT = (
    "This image shows field '8.' of an Uzbekistan driving licence.\n"
    "Field 8 contains a residential address.\n\n"
    "Copy the COMPLETE address:\n"
    "  - There may be TWO or more lines of text\n"
    "  - Read ALL lines from top to bottom\n"
    "  - Join lines with ', '\n"
    "  - Copy every letter, number, comma, and slash exactly as printed\n"
    "  - Pay attention to letter ORDER — do not swap or skip letters\n"
    "  - 'CH' is ALWAYS two separate letters C and H — never write 'Č'\n"
    "  - Read long words fully letter by letter: e.g. Y-U-Q-O-R-I-C-H-I-R-C-H-I-Q\n\n"
    "Return a JSON object with exactly this key: residence"
)

_BACK_PROMPT = (
    "This is the BACK side of an Uzbekistan driving licence.\n\n"
    "There is a table with columns 9, 10, 11, 12.\n"
    "Read EVERY row that contains dates:\n"
    "  col 9  → category  (letter code: A, B, C, D, BE, CE, DE ...)\n"
    "  col 10 → date_of_issue  DD.MM.YYYY — the EARLIER date (issue year)\n"
    "  col 11 → date_of_expiry DD.MM.YYYY — the LATER date (~10 yrs after issue)\n"
    "  col 12 → additional_info — null if cell is empty or '-'\n\n"
    "IMPORTANT: date_of_issue (col 10) year is always SMALLER than date_of_expiry.\n"
    "Read year digits carefully — 2026 and 2036 are different numbers.\n\n"
    "Return a JSON object with exactly one key `category_details`, a list of rows:\n"
    "{\"category_details\": [{\"category\": \"B\", \"date_of_issue\": \"10.04.2026\", "
    "\"date_of_expiry\": \"10.04.2036\", \"additional_info\": null}]}"
)


# ---------------------------------------------------------------------------
# Yordamchi sxemalar
# ---------------------------------------------------------------------------

class _BackData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    category_details: list[CategoryDetail] = Field(default_factory=list)


class _F3Data(BaseModel):
    model_config = ConfigDict(extra="ignore")
    place_of_birth: str | None = None


class _F4Data(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date_of_issue: str | None = None
    date_of_expiry: str | None = None
    place_of_issue: str | None = None
    pinfl: str | None = None


class _F8Data(BaseModel):
    model_config = ConfigDict(extra="ignore")
    residence: str | None = None


# ---------------------------------------------------------------------------
# Krop chegaralari
# ---------------------------------------------------------------------------
#
# Guvohnomaning taxminiy joylashuvi (balandlik %):
#   0-12%   : sarlavha (HAYDOVCHILIK GUVOHNOMASI)
#   12-25%  : 1-maydon (familiya)
#   25-38%  : 2-maydon (ism, 2 qator)
#   36-45%  : 3-maydon (tug'ilgan joy + sana)    ← F3_CROP
#   46-65%  : 4a, 4b, 4c maydonlar               ← F4_CROP (yuqori)
#   60-73%  : 4d maydon (PINFL) va 5-maydon      ← F4_CROP (pastki)
#   70-93%  : 8-maydon (yashash joyi)             ← F8_CROP
#
# Eni bo'yicha:
#   0-22%   : rasm (foto)
#   22-70%  : asosiy matn maydonlari
#   68-100% : o'ng tomonidagi viloyatlar ro'yxati  ← KESIB TASHLANADI
#
# right_pct=0.74 — barcha matnni ushlab turadi (0.68 matn o'rtasini kesardi)
# right_pct=0.64 — faqat f8 uchun (viloyatlar ro'yxatini batamom kesib tashlash)

_F3_TOP, _F3_BOT = 0.37, 0.56   # field 3: tug'ilgan joy + sana (2-maydon ostida, 4a ustida)
_F4_TOP, _F4_BOT = 0.45, 0.82   # fields 4a, 4b, 4c, 4d (3-maydon ostidan boshlanadi)
_F8_TOP, _F8_BOT = 0.70, 0.98   # field 8: yashash joyi — 0.72 dan birinchi qatorni ham qamrab oladi
_RIGHT_F3F4 = 0.85              # f3, f4: 4b sana va 4c matn to'liq ko'rinadi
_RIGHT_F8   = 0.85              # f8: uzun manzil (JIZZAX VILOYATI, FORISH...) to'liq ko'rinadi


# ---------------------------------------------------------------------------
# Token byudjeti
# ---------------------------------------------------------------------------
#
# Qwen3-VL javobdan OLDIN o'ylaydi (`thinking`), JSON esa o'ylashdan KEYIN
# yoziladi. Byudjet kam bo'lsa javob o'ylashning o'rtasida uziladi
# (done_reason='length'), `content` bo'sh qoladi va hamma maydon null chiqadi.
#
# O'lchov (15.08.2026, qwen3-vl:8b, old tomon so'rovi):
#   num_predict=1024 -> done_reason='length', 1024 token o'ylashga ketdi,
#                       content 0 belgi, JSON yo'q
#   num_predict=8192 -> done_reason='stop',  atigi 163 token yozildi,
#                       11 ta maydon to'ldi
#
# Ya'ni chegarani ko'tarish sekinlashtirmaydi: `num_predict` — maqsad emas,
# shift; model ishini tugatsa o'zi to'xtaydi. Kichik byudjet esa faqat
# javobni uzib qo'yadi. Shu sababli krop so'rovlariga ham alohida kichik
# limit berilmaydi — ular ham xuddi shunday uzilib qolardi.
_NUM_PREDICT_MIN = 8192

# Kontekst rasm + prompt + generatsiyani sig'dirishi shart. Rasmning o'zi
# ~2000 token; num_ctx < num_predict bo'lsa model kontekst tugab uziladi.
_NUM_CTX_MIN = 16384


# ---------------------------------------------------------------------------
# Ajratish funksiyasi
# ---------------------------------------------------------------------------

def extract_driving_license(
    front_bytes: bytes,
    back_bytes: bytes | None,
    *,
    ollama_host: str,
    ollama_model: str,
    timeout_seconds: int,
    num_ctx: int,
    num_predict: int,
    ngrok_skip: bool,
) -> tuple[DrivingLicenseData, list[str]]:
    """
    5 ta parallel so'rov:
      front  — to'liq rasm (surname, given_names, date_of_birth, license_number, categories)
      f3     — field 3 krop (place_of_birth)
      f4     — fields 4a-4d krop (date_of_issue, date_of_expiry, place_of_issue, pinfl)
      f8     — field 8 krop (residence) — o'ng tomon kesib tashlangan
      back   — orqa tomon (category_details)
    """

    # `ollama_host` va `ngrok_skip` interfeys mosligi uchun saqlangan, ammo
    # endi ishlatilmaydi: manzil, sarlavhalar va backend tanlovi `ocr.llm`
    # ichida, bitta joyda hal qilinadi (Ollama yoki vLLM).
    base_options: dict[str, Any] = {
        "temperature": 0,
        "seed": 42,
        "num_ctx": max(num_ctx, _NUM_CTX_MIN),
        "num_predict": max(num_predict, _NUM_PREDICT_MIN),
    }

    def _chat_content(
        prompt: str,
        image_path: Path,
        options: dict[str, Any],
    ) -> str:
        """Ollama chat + bo'sh, uzilgan yoki JSON'siz javoblarda qayta urinish.

        5 ta parallel vision so'rov modelni haddan tashqari yuklaydi va
        vaqti-vaqti bilan "Server disconnected" yoki bo'sh javob qaytaradi.
        Bunday o'tkinchi xatolarda qisqa kutib qayta urinamiz.

        Model `content` ni bo'sh qoldirib butun javobni `thinking` ga yozib
        yuborishi ham mumkin — u holda qaytgan matn sof reasoning bo'ladi va
        parse qilinmaydi. Buni xato deb qarab qayta urinamiz, aks holda
        krop natijasi sababsiz yo'qolardi.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                # Prefill (yopiq <think> bloki + '{') FAQAT Ollama yo'lida
                # yuboriladi va javob boshiga qo'shib qaytariladi; vLLM'da
                # o'rniga JSON rejimi ishlaydi. Farqni `vision_chat` biladi.
                content = _strip_think(vision_chat(
                    prompt,
                    [image_path],
                    system=_SYSTEM,
                    prefill=_NO_THINK + "{",
                    json_mode=True,
                    options=options,
                    seed=_seed_for(attempt),
                    timeout=timeout_seconds,
                ))
                if not _find_json(content):
                    raise ValueError("Javobda JSON topilmadi (model faqat o'yladi).")
                return content
            except Exception as exc:  # noqa: BLE001 — o'tkinchi xatolarni ushlaymiz
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    warnings: list[str] = []

    front_path = _save_temp(front_bytes)
    f3_path    = _save_temp(front_bytes, top_pct=_F3_TOP, bottom_pct=_F3_BOT, right_pct=_RIGHT_F3F4)
    f4_path    = _save_temp(front_bytes, top_pct=_F4_TOP, bottom_pct=_F4_BOT, right_pct=_RIGHT_F3F4)
    f8_path    = _save_temp(front_bytes, top_pct=_F8_TOP, bottom_pct=_F8_BOT, right_pct=_RIGHT_F8)
    back_path  = _save_temp(back_bytes) if back_bytes else None

    try:
        def _call_front() -> DrivingLicenseData:
            content = _chat_content(_FRONT_PROMPT, front_path, base_options)
            return _parse(content, warnings)

        def _call_f3() -> _F3Data:
            content = _chat_content(_F3_PROMPT, f3_path, base_options)
            return _parse_model(content, _F3Data, warnings, "Tug'ilgan joy kropi (3)")

        def _call_f4() -> _F4Data:
            content = _chat_content(_F4_PROMPT, f4_path, base_options)
            return _parse_model(content, _F4Data, warnings, "Sanalar kropi (4a-4d)")

        def _call_f8() -> _F8Data:
            content = _chat_content(_F8_PROMPT, f8_path, base_options)
            return _parse_model(content, _F8Data, warnings, "Yashash joyi kropi (8)")

        def _call_back() -> _BackData:
            content = _chat_content(_BACK_PROMPT, back_path, base_options)
            return _parse_back(content, warnings)

        front_data   = DrivingLicenseData()
        f3_result:   _F3Data  | None = None
        f4_result:   _F4Data  | None = None
        f8_result:   _F8Data  | None = None
        back_result: _BackData | None = None

        # Konkurentlikni cheklaymiz: bir vaqtda ko'p vision so'rov modelni
        # yuklab, "Server disconnected" / bo'sh javoblarga sabab bo'ladi.
        # Pool ortiqcha vazifalarni navbatga qo'yib, yukning cho'qqisini pasaytiradi.
        total = 5 if back_path else 4
        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, total)) as pool:
            futures: dict[str, Any] = {
                "front": pool.submit(_call_front),
                "f3":    pool.submit(_call_f3),
                "f4":    pool.submit(_call_f4),
                "f8":    pool.submit(_call_f8),
            }
            if back_path:
                futures["back"] = pool.submit(_call_back)

            for key, future in futures.items():
                try:
                    r = future.result()
                    if key == "front":
                        front_data = r
                    elif key == "f3":
                        f3_result = r
                    elif key == "f4":
                        f4_result = r
                    elif key == "f8":
                        f8_result = r
                    else:
                        back_result = r
                except Exception as exc:
                    warnings.append(f"{key} so'rovda xatolik: {exc}")

        # Krop natijalari to'liq rasm natijasidan USTUN
        overrides: dict[str, Any] = {}

        # place_of_birth: f3 krop (focused) > front rasm
        if f3_result and f3_result.place_of_birth:
            validated_pob = _validate_place_of_birth(f3_result.place_of_birth)
            if validated_pob:
                overrides["place_of_birth"] = validated_pob

        if f4_result:
            if f4_result.date_of_issue:
                overrides["date_of_issue"] = f4_result.date_of_issue
            if f4_result.date_of_expiry:
                overrides["date_of_expiry"] = f4_result.date_of_expiry
            if f4_result.place_of_issue:
                # Krop qiymati ham tekshiruvdan o'tsin: u yerda bosh harfga
                # keltirish va "bitta so'z" filtri bor. Ilgari bu qiymat
                # to'g'ridan-to'g'ri yozilib, ikkalasini ham chetlab o'tardi.
                validated_poi = _validate_place_of_issue(f4_result.place_of_issue)
                if validated_poi:
                    overrides["place_of_issue"] = validated_poi
            if f4_result.pinfl:
                cleaned = _clean_pinfl(f4_result.pinfl)
                if _valid_pinfl(cleaned):
                    overrides["pinfl"] = cleaned

        if f8_result and f8_result.residence:
            clean_res = _validate_residence(f8_result.residence)
            if clean_res:
                overrides["residence"] = clean_res

        if overrides:
            front_data = front_data.model_copy(update=overrides)

        # PINFL
        if front_data.pinfl:
            cleaned = _clean_pinfl(front_data.pinfl)
            front_data = front_data.model_copy(update={
                "pinfl": cleaned if _valid_pinfl(cleaned) else None
            })

        # place_of_birth: f3 krop natijalari allaqachon overrides ga kiritildi;
        # front rasm natijasini faqat krop muvaffaqiyatsiz bo'lganda validate qilamiz
        if front_data.place_of_birth and "place_of_birth" not in overrides:
            front_data = front_data.model_copy(update={
                "place_of_birth": _validate_place_of_birth(front_data.place_of_birth)
            })

        if front_data.place_of_issue and "place_of_issue" not in overrides:
            front_data = front_data.model_copy(update={
                "place_of_issue": _validate_place_of_issue(front_data.place_of_issue)
            })

        # residence: f8 krop muvaffaqiyatsiz bo'lganda front rasmga ham validate qo'llaymiz
        if front_data.residence and "residence" not in overrides:
            front_data = front_data.model_copy(update={
                "residence": _validate_residence(front_data.residence)
            })

        if back_result:
            fixed = _fix_category_date_order(back_result.category_details)
            front_data = front_data.model_copy(update={"category_details": fixed})

    finally:
        front_path.unlink(missing_ok=True)
        f3_path.unlink(missing_ok=True)
        f4_path.unlink(missing_ok=True)
        f8_path.unlink(missing_ok=True)
        if back_path:
            back_path.unlink(missing_ok=True)

    return front_data, warnings


# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------

_MAX_SIDE = 1600

# O'tkinchi Ollama xatolari (uzilish / bo'sh javob) uchun qayta urinish.
# 3 emas, 2: so'rov `temperature=0, seed=42` bilan determinlashgan, ya'ni
# bir xil urug' bilan qayta urinish AYNAN o'sha javobni qaytaradi. Shuning
# uchun har urinishda urug' o'zgartiriladi (`_seed_for`) — aks holda uchala
# urinish bir xil xatoni takrorlab, faqat vaqt yeydi.
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.5

# Bir vaqtda modelga yuboriladigan maksimal parallel vision so'rovlar.
# Ilgari 2 edi — 5 ta so'rov 3 to'lqinga bo'linib, vaqt uch barobar cho'zilardi.
# Ollama 0.20 parallel so'rovlarni o'zi navbatga qo'yadi, shuning uchun
# hammasini birdan yuboramiz.
_MAX_CONCURRENCY = 5


def _seed_for(attempt: int) -> int:
    """Har urinishga boshqa urug' — takroriy so'rov boshqa natija bersin."""
    return 42 + attempt * 1000


def _save_temp(
    image_bytes: bytes,
    *,
    top_pct: float = 0.0,
    bottom_pct: float = 1.0,
    left_pct: float = 0.0,
    right_pct: float = 1.0,
) -> Path:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    box = (int(w * left_pct), int(h * top_pct), int(w * right_pct), int(h * bottom_pct))
    if box != (0, 0, w, h):
        img = img.crop(box)
        w, h = img.size
    if max(w, h) > _MAX_SIDE:
        scale = _MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    with NamedTemporaryFile(suffix=".jpg", prefix="dl_ocr_", delete=False) as f:
        path = Path(f.name)
        img.save(f, format="JPEG", quality=92, optimize=True)
    return path


def _clean_pinfl(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def _valid_pinfl(value: str) -> bool:
    return len(_clean_pinfl(value)) == 14


_REGION_LIST_WORDS = {
    "QORAQALPOG'ISTON", "ANDIJON", "BUXORO", "JIZZAX", "QASHQADARYO",
    "NAVOIY", "NAMANGAN", "SAMARQAND", "SIRDARYO", "SURXONDARYO",
    "FARG'ONA", "XORAZM",
}

# Guvohnomadagi matn maydonlari butunlay BOSH HARFDA bosilgan, model esa
# ba'zan aralash yozadi: "VILoyATI", "ToSHKENT". Quyidagi tekshiruvchilar
# ilgari `upper()` ni faqat solishtirish uchun ishlatib, natijaga asl matnni
# qaytarardi — shuning uchun aralash harflar javobga o'tib ketardi.
# Texpasportda bu holat allaqachon hal qilingan (`_fix_place_words`).


def _validate_residence(value: str) -> str | None:
    """Viloyatlar ro'yxatidan o'qilgan yolg'on qiymatlarni filterlaydi."""
    # Model CH → Č yoki ch → č yozishi mumkin; normallashtirish
    value = value.replace("Č", "CH").replace("č", "CH")
    upper = value.upper().strip()
    # "VILOYAT" ildizi VILOYATI ham, VILAYATI (model yozish xatosi) ham ushlab qoladi
    region_word_count = upper.count("VILOYAT") + upper.count("RESPUBLIKASI")
    if region_word_count >= 2:
        return None
    if upper.count("/") >= 3:
        return None
    return upper or None


def _validate_place_of_birth(value: str) -> str | None:
    """Region ro'yxatidan o'qilgan tug'ilgan joy qiymatlarini filterlaydi."""
    upper = value.upper().strip()
    # "JIZZAX VILOYATI" kabi — viloyat nomi, shahar/tuman emas
    if upper.endswith("VILOYATI") or upper.endswith("VILAYATI") or upper.endswith("RESPUBLIKASI"):
        return None
    return upper or None


def _validate_place_of_issue(value: str) -> str | None:
    """Bitta so'zdan iborat buzilgan qiymatlarni filterlaydi."""
    upper = value.upper().strip()
    # Haqiqiy qiymat kamida 2 so'z: "YUNUSOBOD TUMANI DXM", "TOSHKENT SHAHRI IIB"
    if len(upper.split()) < 2:
        return None
    return upper


def _fix_category_date_order(details: list[CategoryDetail]) -> list[CategoryDetail]:
    result = []
    for d in details:
        if d.date_of_issue and d.date_of_expiry:
            try:
                issue_dt  = datetime.strptime(d.date_of_issue, "%d.%m.%Y")
                expiry_dt = datetime.strptime(d.date_of_expiry, "%d.%m.%Y")
                if issue_dt > expiry_dt:
                    d = d.model_copy(update={
                        "date_of_issue": d.date_of_expiry,
                        "date_of_expiry": d.date_of_issue,
                    })
            except (ValueError, TypeError):
                pass
        result.append(d)
    return result




def _field(source: Any, *names: str) -> str | None:
    """Dict yoki obyektdan birinchi bo'sh bo'lmagan matn maydonini qaytaradi."""
    if source is None:
        return None
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# qwen3-vl "thinking" modeli: erkin qo'yilsa minglab token o'ylab, javobga
# yetib bormaydi (done_reason='length', content bo'sh). `think=False` ham,
# `/nothink` ham bu versiyada e'tiborga olinmaydi — o'lchangan: bitta krop
# so'rovi 41 s ketkazib, 8192 tokenni to'liq yeb, JSON qaytarmadi.
# Yagona ishlaydigan yo'l — assistant javobini yopiq <think> bloki bilan
# oldindan to'ldirish. Xuddi shu yechim `ocr/id_card/extractor.py` da ham.
_NO_THINK = "<think>\n\n</think>\n\n"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_think(content: str) -> str:
    """Model yozib yuborgan reasoning blokini olib tashlaydi.

    Yopilgan `<think>...</think>` butunlay o'chiriladi. Blok yopilmagan
    bo'lsa (javob uzilgan), undan keyin JSON ham yo'q — qoldig'i kesiladi.
    """
    cleaned = _THINK_BLOCK.sub("", content)
    if "<think>" in cleaned.lower():
        cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


_T = Any


def _parse_model(content: str, model_cls: type[_T], warnings: list[str], label: str) -> _T:
    """Krop so'rovi javobini o'qiydi.

    Ilgari xatolik jim yutilar edi — krop uzilib qolsa javobda maydon
    yo'qolardi, lekin sababi hech qayerda ko'rinmasdi. Endi ogohlantirish
    yoziladi: `front` uchun yozilgani bilan bir xil mantiq.
    """
    for candidate in (content, _find_json(content)):
        if not candidate:
            continue
        try:
            return model_cls.model_validate_json(candidate)
        except Exception:
            pass
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return model_cls.model_validate(payload)
        except Exception:
            pass
    snippet = content.strip().replace("\n", " ")[:120]
    warnings.append(f"{label}: model javobi parse qilinmadi. Javob: {snippet!r}")
    return model_cls()


def _parse(content: str, warnings: list[str]) -> DrivingLicenseData:
    for candidate in (content, _find_json(content)):
        if not candidate:
            continue
        try:
            return DrivingLicenseData.model_validate_json(candidate)
        except Exception:
            pass
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return DrivingLicenseData.model_validate(payload)
        except Exception:
            pass
    warnings.append("Model javobi to'liq parse qilinmadi; qisman natija qaytarildi.")
    return DrivingLicenseData()


def _parse_back(content: str, warnings: list[str]) -> _BackData:
    for candidate in (content, _find_json(content)):
        if not candidate:
            continue
        try:
            return _BackData.model_validate_json(candidate)
        except Exception:
            pass
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return _BackData.model_validate(payload)
            if isinstance(payload, list):
                return _BackData(
                    category_details=[CategoryDetail.model_validate(r) for r in payload]
                )
        except Exception:
            pass
    warnings.append("Orqa tomon jadvali parse qilinmadi.")
    return _BackData()


def _find_json(text: str) -> str | None:
    """Matndan JSON obyektini ajratadi.

    Prefill (`{`) tufayli javob yopilmagan bo'lishi mumkin — model oxirgi
    qavsni yozmasdan to'xtaydi. Shuning uchun yopilmagan variantni ham
    tiklashga urinamiz (ID-karta modulidagi `_parse_json` bilan bir mantiq).
    """
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except (TypeError, ValueError):
            pass
    if start < 0:
        return None
    head = text[start:].rstrip().rstrip(",")
    for suffix in ("}", '"}', "]}", '"}]}'):
        candidate = head + suffix
        try:
            json.loads(candidate)
            return candidate
        except (TypeError, ValueError):
            continue
    return None
