"""O'zbekiston texpasporti — Ollama vision orqali ma'lumot ajratish.

Hujjat: "AVTOMOTOTRANSPORT VOSITASI RO'YXATDAN O'TKAZILGANLIGI TO'G'RISIDA GUVOHNOMA".
Old tomon — 1..8 maydonlar, orqa tomon — 9..19 maydonlar va seriya raqami.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from PIL import Image
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from ocr.llm import vision_chat


# ---------------------------------------------------------------------------
# Sxemalar
# ---------------------------------------------------------------------------

def _to_text(value: Any) -> Any:
    """Model qaytargan har qanday JSON turini matnga keltiradi.

    Vision model sxemaga qaramay son (`"seats": 2`, `"year_of_manufacture": 1984`)
    yoki ko'p qatorli manzilni ro'yxat qilib qaytaradi. Pydantic sukut bo'yicha
    int/list ni str ga o'girmaydi — bitta shunday maydon butun javobni yiqitardi.
    """
    if value is None or isinstance(value, str):
        text = value.strip() if isinstance(value, str) else None
        return text or None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if item not in (None, "")]
        return ", ".join(parts) or None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


LooseStr = Annotated[str | None, BeforeValidator(_to_text)]


class TexPasportData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Old tomon
    plate_number: LooseStr = Field(None, description="Davlat raqam belgisi (1)")
    make_model: LooseStr = Field(None, description="Rusumi / modeli (2)")
    color: LooseStr = Field(None, description="Rangi (3)")
    owner: LooseStr = Field(None, description="Egasi — F.I.O. yoki korxona nomi (4)")
    address: LooseStr = Field(None, description="Manzili — to'liq yashash joyi (5)")
    date_of_issue: LooseStr = Field(None, description="Berilgan sanasi DD.MM.YYYY (6)")
    issuing_authority: LooseStr = Field(None, description="YHX boshqarmasi bo'limi (7)")
    pinfl_tin: LooseStr = Field(None, description="JSHSHIR / STIR — raqamlar (8)")

    # Orqa tomon
    year_of_manufacture: LooseStr = Field(None, description="Ishlab chiqarilgan yili (9)")
    vehicle_type: LooseStr = Field(None, description="Turi — YUK FURGON, YENGIL AVTOMOBIL ... (10)")
    vin: LooseStr = Field(None, description="Kuzov / shassi raqami (11)")
    max_mass_kg: LooseStr = Field(None, description="To'la vazni, kg (12)")
    unladen_mass_kg: LooseStr = Field(None, description="Yuksiz vazni, kg (13)")
    engine_number: LooseStr = Field(None, description="Dvigatel raqami (14)")
    engine_capacity: LooseStr = Field(None, description="Dvigatel quvvati (15)")
    fuel_type: LooseStr = Field(None, description="Yonilg'i turi — BENZIN, DIZEL, GBASPG ... (16)")
    seats: LooseStr = Field(None, description="O'tiradigan joylar soni, haydovchi bilan (17)")
    standing_places: LooseStr = Field(None, description="Tik turadigan joylar soni (18)")
    special_marks: LooseStr = Field(None, description="Alohida belgilar (19)")


class TexPasportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = Field("success", description="'success' yoki 'partial'")
    model: str = ""
    data: TexPasportData = Field(default_factory=TexPasportData)
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

# Promptlar ataylab QISQA: model har bir ko'rsatma ustida "o'ylaydi" (reasoning),
# uzun prompt esa javob vaqtini bir necha barobar oshiradi.

_FRONT_PROMPT = (
    "Uzbekistan vehicle registration certificate, FRONT side. Copy each numbered value:\n"
    "1 → plate_number, 2 → make_model, 3 → color, 4 → owner, 5 → address (join lines with ', '),\n"
    "6 → date_of_issue (DD.MM.YYYY), 7 → issuing_authority, 8 → pinfl_tin (digits).\n"
    "Ignore the preprinted region list on the right edge. Null if unreadable."
)

_ADDRESS_PROMPT = (
    "Uzbekistan vehicle registration certificate, cropped. Copy two values:\n"
    "4 → owner (person or company), 5 → address (2-3 lines, join with ', ').\n"
    "Ignore the preprinted region list on the right edge. Null if unreadable."
)

_BACK_PROMPT = (
    "Uzbekistan vehicle registration certificate, BACK side. Copy each numbered value:\n"
    "9 → year_of_manufacture, 10 → vehicle_type, 11 → vin, 12 → max_mass_kg (number),\n"
    "13 → unladen_mass_kg (number), 14 → engine_number, 15 → engine_capacity,\n"
    "16 → fuel_type, 17 → seats, 18 → standing_places, 19 → special_marks.\n"
    "Cyrillic text: transliterate to Latin (РАКАМСИЗ → RAKAMSIZ, ҚАРОР → QAROR).\n"
    "Ignore the preprinted region list on the right edge. Null if unreadable."
)


# ---------------------------------------------------------------------------
# Yordamchi sxemalar
# ---------------------------------------------------------------------------

class _AddressData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    owner: LooseStr = None
    address: LooseStr = None


class _FrontData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    plate_number: LooseStr = None
    make_model: LooseStr = None
    color: LooseStr = None
    owner: LooseStr = None
    address: LooseStr = None
    date_of_issue: LooseStr = None
    issuing_authority: LooseStr = None
    pinfl_tin: LooseStr = None


class _BackData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    year_of_manufacture: LooseStr = None
    vehicle_type: LooseStr = None
    vin: LooseStr = None
    max_mass_kg: LooseStr = None
    unladen_mass_kg: LooseStr = None
    engine_number: LooseStr = None
    engine_capacity: LooseStr = None
    fuel_type: LooseStr = None
    seats: LooseStr = None
    standing_places: LooseStr = None
    special_marks: LooseStr = None


# ---------------------------------------------------------------------------
# Token byudjeti
# ---------------------------------------------------------------------------
#
# Model reasoning ("<think> ...") yozsa, JSON undan KEYIN keladi. Byudjet kam
# bo'lsa javob <think> ning o'rtasida uzilib qoladi va hamma maydon null chiqadi.
# .env dagi MODEL_NUM_PREDICT shu chegaralar orasiga qisiladi.

# O'lchangan qiymat: 8 maydonli old tomon uchun model 2048 tokenni to'liq sarflab
# ham reasoning ni tugatmadi (done_reason='length'). JSON reasoning dan KEYIN keladi,
# shuning uchun byudjet reasoning + JSON ga yetishi kerak.
_NUM_PREDICT_MIN = 8192
_NUM_PREDICT_MAX = 8192
_NUM_PREDICT_BACK = 8192   # orqa tomon: 11 ta maydon — eng uzun reasoning

# Kontekst rasm + prompt + generatsiyani sig'dirishi shart. num_ctx < num_predict
# bo'lsa, model kontekst tugab, javobni baribir uzib qo'yadi.
_NUM_CTX = 16384


# ---------------------------------------------------------------------------
# Krop chegaralari
# ---------------------------------------------------------------------------
#
# Old tomonning taxminiy joylashuvi (balandlik %):
#   0-14%   : sarlavha
#   14-30%  : 1, 2-maydonlar (davlat raqami, rusumi)
#   30-45%  : 3, 4-maydonlar (rangi, egasi)
#   40-72%  : 5-maydon (manzil, 2-3 qator)     ← ADDR_CROP
#   72-100% : 6, 7, 8-maydonlar
#
# Eni bo'yicha o'ng chekkada viloyatlar ro'yxati bosilgan (preprinted) —
# u modelni chalg'itadi, shuning uchun kroplarda kesib tashlanadi.

_ADDR_TOP, _ADDR_BOT = 0.38, 0.68   # 4 va 5-maydonlar
_RIGHT_FRONT = 0.82                 # o'ng chekkadagi viloyatlar ro'yxatini kesish
_RIGHT_ADDR = 0.80                  # manzil krop — ro'yxatni batamom kesish


# ---------------------------------------------------------------------------
# Ajratish funksiyasi
# ---------------------------------------------------------------------------

def extract_texpasport(
    front_bytes: bytes,
    back_bytes: bytes | None,
    *,
    ollama_host: str,
    ollama_model: str,
    timeout_seconds: int,
    num_ctx: int,
    num_predict: int,
    ngrok_skip: bool,
) -> tuple[TexPasportData, list[str]]:
    """
    3 ta parallel so'rov:
      front  — old tomon (1..8 maydonlar), o'ng chekka kesilgan
      addr   — 4 va 5-maydonlar kropi (owner, address) — front natijasidan ustun
      back   — orqa tomon (9..19 maydonlar + seriya raqami)
    """

    headers: dict[str, str] = {}
    if ngrok_skip and "ngrok" in ollama_host.lower():
        headers["ngrok-skip-browser-warning"] = "true"

    # `client` endi kerak emas: manzil, sarlavha va backend tanlovi
    # `ocr.llm.vision_chat` ichida, bitta joyda hal qilinadi.
    # Model reasoning ("<think> ...") yozib yuborsa, JSON undan keyin keladi —
    # shuning uchun token byudjeti reasoning + JSON ga yetarli bo'lishi shart.
    # Kam num_predict javobni <think> ning o'rtasida uzib qo'yadi.
    base_options: dict[str, Any] = {
        "temperature": 0,
        "seed": 42,
        "num_ctx": max(num_ctx, _NUM_CTX),
        "num_predict": max(min(num_predict, _NUM_PREDICT_MAX), _NUM_PREDICT_MIN),
    }
    # Orqa tomonda 12 ta maydon bor — model har biri uchun reasoning yozsa,
    # oddiy byudjet JSON ga yetmay uziladi. Shuning uchun alohida, kattaroq limit.
    back_options: dict[str, Any] = {**base_options, "num_predict": _NUM_PREDICT_BACK}

    def _chat_content(
        prompt: str,
        image_path: Path,
        format_schema: dict[str, Any],
        options: dict[str, Any],
    ) -> str:
        """Ollama chat: bo'sh, uzilgan yoki JSON'siz javoblarda qayta urinadi."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                content = _strip_think(vision_chat(
                    _no_think(prompt),
                    [image_path],
                    system=_SYSTEM,
                    json_schema=format_schema,
                    options=options,
                    timeout=timeout_seconds,
                ))
                if not _find_json(content):
                    raise ValueError(
                        f"Javobda JSON topilmadi. Javob boshi: {content[:200]!r}"
                    )
                return content
            except Exception as exc:  # noqa: BLE001 — o'tkinchi xatolarni ushlaymiz
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    warnings: list[str] = []

    front_path = _save_temp(front_bytes, right_pct=_RIGHT_FRONT)
    addr_path = _save_temp(
        front_bytes, top_pct=_ADDR_TOP, bottom_pct=_ADDR_BOT, right_pct=_RIGHT_ADDR
    )
    back_path = _save_temp(back_bytes) if back_bytes else None

    try:
        def _call_front() -> _FrontData:
            content = _chat_content(
                _FRONT_PROMPT, front_path,
                _FrontData.model_json_schema(),
                base_options,
            )
            return _parse_model(content, _FrontData, warnings, "Old tomon")

        def _call_addr() -> _AddressData:
            content = _chat_content(
                _ADDRESS_PROMPT, addr_path,
                _AddressData.model_json_schema(),
                base_options,
            )
            return _parse_model(content, _AddressData, warnings, "Manzil kropi")

        def _call_back() -> _BackData:
            content = _chat_content(
                _BACK_PROMPT, back_path,
                _BackData.model_json_schema(),
                back_options,
            )
            return _parse_model(content, _BackData, warnings, "Orqa tomon")

        front_data = _FrontData()
        addr_result: _AddressData | None = None
        back_result: _BackData | None = None

        # Uchala so'rov bir-biriga bog'liq emas — parallel ketadi (konkurentlik cheklangan).
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY) as pool:
            futures: dict[str, Any] = {
                "front": pool.submit(_call_front),
                "addr": pool.submit(_call_addr),
            }
            if back_path:
                futures["back"] = pool.submit(_call_back)

            for key, future in futures.items():
                try:
                    result = future.result()
                    if key == "front":
                        front_data = result
                    elif key == "addr":
                        addr_result = result
                    else:
                        back_result = result
                except Exception as exc:
                    warnings.append(f"{key} so'rovda xatolik: {exc}")

        payload: dict[str, Any] = front_data.model_dump()

        # Manzil kropi to'liq rasmdan aniqroq o'qiydi (qishloq/mahalla nomlari),
        # shuning uchun har doim yuboriladi va natijasi front dan USTUN turadi.
        if addr_result:
            cleaned = _validate_address(addr_result.address) if addr_result.address else None
            if cleaned:
                payload["address"] = cleaned
            if addr_result.owner:
                payload["owner"] = addr_result.owner

        if back_result:
            for key, value in back_result.model_dump().items():
                if value:
                    payload[key] = value

        data = TexPasportData.model_validate(_normalize(payload))

    finally:
        front_path.unlink(missing_ok=True)
        if addr_path:
            addr_path.unlink(missing_ok=True)
        if back_path:
            back_path.unlink(missing_ok=True)

    return data, warnings


# ---------------------------------------------------------------------------
# Normallashtirish
# ---------------------------------------------------------------------------

_REGION_MARKERS = ("VILOYATI", "VILAYATI", "RESPUBLIKASI")

# Hujjatda ba'zi so'zlar kirillda bosilgan (РАКАМСИЗ, ҚАРОР, АЛМАШТИРИШ).
_CYRILLIC_TO_LATIN = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Ғ": "G'", "Д": "D", "Е": "E", "Ё": "YO",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Қ": "Q", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ў": "O'",
    "Ф": "F", "Х": "X", "Ҳ": "H", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SH",
    "Ъ": "'", "Ы": "I", "Ь": "", "Э": "E", "Ю": "YU", "Я": "YA", "№": "N",
}

# Model kirillni lotinga o'girmay, ko'rinishi o'xshash harflar bilan ko'chirib yozadi:
#   РАКАМСИЗ → "PAKAMCSI3",  ҚАРОР → "KAPOP" / "KAPOR"
# Shu sababli buzilgan so'zlarni o'xshashlik bo'yicha tiklaymiz.
_KNOWN_WORDS = ("RAKAMSIZ", "QAROR", "ALMASHTIRISH")
_SIMILARITY_THRESHOLD = 0.78
# Faqat kirill bosiladigan maydonlar — ism/manzilni tasodifan buzib yubormaslik uchun
_CYRILLIC_PRONE_FIELDS = ("vin", "engine_number", "special_marks", "vehicle_type", "fuel_type")

# Model kirill harfni shakli o'xshash lotin harf/raqam bilan almashtiradi.
# Taqqoslashdan OLDIN teskari o'giramiz (faqat solishtirish uchun, natijaga yozilmaydi).
_LOOKALIKE = str.maketrans({"P": "R", "C": "S", "3": "Z", "0": "O", "1": "I", "4": "A", "5": "S"})


# Joy nomlari lug'ati — model Q ni K/G deb o'qiydi (SAMARQAND → SAMARKAND / SAMARGAND),
# X ni H deb o'qiydi. Taqqoslashda shu harflarni bir xil deb qaraymiz, shunda
# buzilgan variant lug'atdagi to'g'ri nom bilan mos tushadi.
_CONFUSABLE = str.maketrans({"Q": "K", "G": "K", "X": "H", "'": "", "‘": "", "’": ""})

# Manzil va berilgan joyda uchraydigan tayanch so'zlar
_PLACE_WORDS = (
    "O'ZBEKISTON", "RESPUBLIKASI", "QORAQALPOG'ISTON",
    "VILOYATI", "TUMANI", "SHAHRI", "SHAHARCHASI", "QISHLOG'I", "MAHALLASI",
    "KO'CHASI", "QFY", "MFY", "IOB", "TRB", "YHX", "UY",
)

# Ixtiyoriy fayl: `{"Viloyat nomi": [{"name": "Shahar"}, ...]}` ko'rinishida.
# Asl loyihada u boshqa modulning ichida turardi; bu repoda yo'q, shuning
# uchun yo'q bo'lsa quyidagi tayanch so'zlar bilan cheklanamiz.
_UZ_CITIES_PATH = Path(__file__).resolve().parent / "uz_cities.json"


def _load_place_vocabulary() -> dict[str, str]:
    """Joy nomlari lug'ati: chalkash-harf kaliti → to'g'ri yozilishi.

    Viloyat va shahar nomlari `uz_cities.json` dan olinadi; fayl bo'lmasa
    faqat `_PLACE_WORDS` dagi tayanch so'zlar to'g'irlanadi.
    """
    vocabulary: dict[str, str] = {}

    def add(word: str) -> None:
        word = word.strip().upper()
        if len(word) >= 3:
            vocabulary.setdefault(word.translate(_CONFUSABLE), word)

    for word in _PLACE_WORDS:
        add(word)

    try:
        raw = json.loads(_UZ_CITIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return vocabulary  # ro'yxat yo'q bo'lsa, tayanch so'zlar bilan davom etamiz

    for region, cities in raw.items():
        for part in region.split():
            add(part)
        for city in cities:
            name = city.get("name") if isinstance(city, dict) else None
            if name:
                add(name)
    return vocabulary


_PLACE_VOCABULARY = _load_place_vocabulary()
# Manzilga o'xshash, joy nomlari uchraydigan maydonlar
_PLACE_FIELDS = ("address", "issuing_authority")


def _fix_place_words(value: str) -> str:
    """Joy nomlaridagi Q/K/G va X/H chalkashuvini lug'at bo'yicha to'g'irlaydi.

    'SAMARKAND' va 'SAMARGAND' → 'SAMARQAND', 'TOSHKEND' → 'TOSHKENT'.
    Lug'atda yo'q so'zlar (qishloq, mahalla nomlari) o'zgarishsiz qoladi.
    """

    def repair(match: re.Match[str]) -> str:
        token = match.group(0)
        upper = token.upper()
        if len(upper) < 3:
            return upper
        exact = _PLACE_VOCABULARY.get(upper.translate(_CONFUSABLE))
        if exact:
            return exact
        return upper  # hujjat butunlay bosh harfda — "VILoyATI" ni ham to'g'rilaydi

    return re.sub(r"[A-Za-z'‘’]+", repair, value)


def _translit(value: str) -> str:
    """Kirill harflarni lotin o'zbek alifbosiga o'giradi."""
    return "".join(_CYRILLIC_TO_LATIN.get(ch, _CYRILLIC_TO_LATIN.get(ch.upper(), ch)) for ch in value)


def _fix_garbled_words(value: str) -> str:
    """Kirilldan noto'g'ri ko'chirilgan so'zlarni tiklaydi: 'PAKAMCSI3' → 'RAKAMSIZ'."""

    def repair(match: re.Match[str]) -> str:
        token = match.group(0)
        upper = token.upper()
        if upper in _KNOWN_WORDS:
            return token
        # Kamida 4 ta harf bo'lsin — sof raqamlar (kuzov/dvigatel raqami) tegilmaydi
        if sum(ch.isalpha() for ch in upper) < 4:
            return token
        key = upper.translate(_LOOKALIKE)
        for word in _KNOWN_WORDS:
            if SequenceMatcher(None, key, word).ratio() >= _SIMILARITY_THRESHOLD:
                return word
        return token

    return re.sub(r"[A-Za-z0-9']+", repair, value)


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Model qaytargan qiymatlarni tozalaydi va formatlarni bir xillashtiradi."""
    result = {k: (v.strip() if isinstance(v, str) else v) for k, v in payload.items()}

    for key, value in list(result.items()):
        if isinstance(value, str):
            # Kirill → lotin, so'ng "- LABO" kabi bosh chiziqchani olib tashlaymiz
            value = _translit(value).lstrip("-").strip()
            if key in _CYRILLIC_PRONE_FIELDS:
                value = _fix_garbled_words(value)
            if key in _PLACE_FIELDS:
                value = _fix_place_words(value)
            result[key] = value or None

    if result.get("plate_number"):
        result["plate_number"] = re.sub(r"[\s-]", "", result["plate_number"]).upper()

    for key in ("pinfl_tin", "year_of_manufacture", "seats", "standing_places"):
        if result.get(key):
            digits = re.sub(r"\D", "", result[key])
            result[key] = digits or None

    for key in ("max_mass_kg", "unladen_mass_kg"):
        if result.get(key):
            result[key] = _clean_mass(result[key])

    if result.get("date_of_issue"):
        result["date_of_issue"] = _normalize_date(result["date_of_issue"])

    if result.get("address"):
        result["address"] = _validate_address(result["address"])

    return result


def _clean_mass(value: str) -> str | None:
    """'17 835.00 (KG)' → '17835.00'."""
    match = re.search(r"\d[\d\s.,]*", value)
    if not match:
        return None
    number = match.group(0).replace(" ", "").replace(",", ".").rstrip(".")
    return number or None


def _normalize_date(value: str) -> str | None:
    """Sanani DD.MM.YYYY ko'rinishiga keltiradi."""
    digits = re.findall(r"\d+", value)
    if len(digits) == 3:
        day, month, year = digits
        if len(year) == 4:
            return f"{day.zfill(2)}.{month.zfill(2)}.{year}"
    return value or None


def _validate_address(value: str) -> str | None:
    """O'ng chekkadagi viloyatlar ro'yxatidan o'qilgan yolg'on qiymatlarni filterlaydi."""
    value = value.replace("Č", "CH").replace("č", "ch").strip()
    upper = value.upper()
    marker_count = sum(upper.count(marker) for marker in _REGION_MARKERS)
    # Haqiqiy manzilda "VILOYATI" bir marta uchraydi; 2+ bo'lsa — ro'yxat o'qilgan
    if marker_count >= 2:
        return None
    return value or None


# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------

# Rasm qancha katta bo'lsa, vision tokenlari shuncha ko'p va prefill shuncha sekin.
# Lekin 1280 px da qishloq/mahalla nomlari (DASHTAKIBOLO, DEHNAV) buzila boshladi —
# aniqlik tezlikdan muhimroq, shuning uchun 1600 px.
_MAX_SIDE = 1600

# O'tkinchi Ollama xatolari (uzilish / bo'sh javob) uchun qayta urinish
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5
# Bir vaqtda modelga yuboriladigan maksimal parallel vision so'rovlar
_MAX_CONCURRENCY = 2


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
    with NamedTemporaryFile(suffix=".jpg", prefix="tex_ocr_", delete=False) as f:
        path = Path(f.name)
        img.save(f, format="JPEG", quality=92, optimize=True)
    return path


def _no_think(prompt: str) -> str:
    """Qwen3 ning soft-switch direktivasi — user xabarida ko'rsa reasoning ni o'chiradi.

    To'g'ri token — `/no_think` (pastki chiziq bilan); `/nothink` e'tiborga olinmaydi.
    """
    return f"/no_think\n{prompt}\n\nRespond with the JSON object only. Do not think out loud."


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_think(content: str) -> str:
    """Model yozib yuborgan reasoning blokini olib tashlaydi.

    Yopilgan `<think>...</think>` butunlay o'chiriladi. Agar blok yopilmagan
    bo'lsa (javob uzilib qolgan), undan keyin JSON ham yo'q — qoldig'i kesiladi.
    """
    cleaned = _THINK_BLOCK.sub("", content)
    if "<think>" in cleaned.lower():
        cleaned = _THINK_OPEN.sub("", cleaned)
    return cleaned.strip()


def _field(source: Any, *names: str) -> str | None:
    """Dict yoki obyektdan birinchi bo'sh bo'lmagan matn maydonini qaytaradi."""
    if source is None:
        return None
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_model(content: str, model_cls: Any, warnings: list[str], label: str) -> Any:
    for candidate in (content, _find_json(content)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if not isinstance(payload, dict):
            continue
        return _lenient_validate(model_cls, payload)

    snippet = content.strip().replace("\n", " ")[:180]
    warnings.append(f"{label}: model javobi parse qilinmadi. Javob: {snippet!r}")
    return model_cls()


def _lenient_validate(model_cls: Any, payload: dict[str, Any]) -> Any:
    """Bitta buzuq maydon butun javobni yo'qotmasligi uchun maydonma-maydon tekshiradi."""
    try:
        return model_cls.model_validate(payload)
    except Exception:
        pass
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in model_cls.model_fields:
            continue
        try:
            model_cls.model_validate({key: value})
        except Exception:
            continue
        clean[key] = value
    return model_cls.model_validate(clean)


_DECODER = json.JSONDecoder()


def _find_json(text: str) -> str | None:
    """Matn ichidan birinchi to'liq JSON obyektini topadi.

    Model JSON oldidan izoh yozishi yoki JSON dan keyin matn qo'shishi mumkin,
    shuning uchun har bir '{' dan boshlab raw_decode bilan urinib ko'ramiz.
    Uzilib qolgan (yopilmagan) JSON hech qayerda dekod bo'lmaydi → None.
    """
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = _DECODER.raw_decode(text, index)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload:
            return text[index:end]
    return None
