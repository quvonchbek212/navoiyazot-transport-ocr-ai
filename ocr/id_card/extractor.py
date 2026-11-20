"""O'zbekiston ID kartasidan barcha ma'lumotlarni ajratib olish.

Ollama'dagi qwen3-vl:8b vision modeli orqali kartaning old va orqa tomonini
o'qiydi, MRZ zonasini checksum bilan tekshiradi va tekshirilgan JSON qaytaradi.

Ishlatish (loyiha ildizidan):
    python -m ocr.id_card.extractor old.jpg
    python -m ocr.id_card.extractor old.jpg --back orqa.jpg -o natija.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from ocr import config
from ocr.llm import vision_chat
from ocr.id_card import mrz as mrz_lib
from ocr.id_card import preprocess, validate

OLLAMA_URL = config.OLLAMA_BASE_URL
MODEL = config.MODEL_NAME
MAX_SIDE = 1600  # MRZ o'qilishi uchun yetarli, lekin modelni bo'kdirmaydi

USE_MRZ = config.USE_MRZ
TIMEOUT = config.TIMEOUT_SECONDS
HEADERS = config.ollama_headers()

# qwen3-vl "thinking" modeli: erkin qo'yilsa minglab token o'ylab, javobga
# yetib bormaydi (done_reason=length, content bo'sh). `think:false` ham,
# `/no_think` ham bu versiyada e'tiborga olinmaydi. Yagona ishlaydigan yo'l -
# assistant javobini yopiq <think> bloki bilan oldindan to'ldirish.
NO_THINK = "<think>\n\n</think>\n\n"

# Diqqat: Ollama'ning `format` (JSON schema) rejimidan ataylab foydalanilmagan.
# Bu model bilan grammatika cheklovi generatsiyani 300s+ ga cho'zib yuboradi
# (ayniqsa ["string","null"] union tiplarida), ba'zan esa model rasmni o'qish
# o'rniga "John Doe" kabi shablon qiymat qaytaradi. Prefill + JSON tahlili
# tezroq (~3s) va aniqroq.

FIELDS = ["document_number", "pinfl", "issue_date", "expiry_date", "issued_by"]

PROMPT = f"""You are an OCR system reading a real Uzbekistan ID card (shaxsni tasdiqlovchi ID-karta).

Transcribe ONLY what is literally printed in the image. Never invent placeholder
values such as "123456789" or "AA1234567" - if a field is not visible, use "".

Field guide (card labels are bilingual Uzbek/English/Russian):
  document_number - "Karta raqami" / "Document No." / "Seriya va raqami".
                    Format: exactly 2 letters + 7 digits, e.g. AD8718879.
                    Usually printed at the TOP RIGHT of the card.
  pinfl           - "JSHSHIR" / "PINFL" / "ЖШШИР". Exactly 14 digits, no letters.
  issue_date      - "Berilgan sanasi" / "Date of issue". The EARLIER date.
  expiry_date     - "Amal qilish muddati" / "Date of expiry". The LATER date,
                    normally exactly 10 years after issue_date.
  issued_by       - "Kim tomonidan berilgan" / issuing authority.
                    Often "IIV" or "IIB" followed by digits, e.g. "IIV 18224".

Critical rules - these are the mistakes to avoid:
- document_number and pinfl are DIFFERENT fields. document_number has letters,
  pinfl is 14 digits only. Never put one into the other.
- issue_date must be EARLIER than expiry_date. Read the labels, do not guess
  from position.
- Dates: convert DD.MM.YYYY to YYYY-MM-DD.
- Do NOT take any digits from the machine-readable zone (the lines full of
  "<<<<" characters at the bottom); that zone is read separately.
- Read each digit carefully one by one. If a digit is genuinely unreadable,
  return "" for that whole field rather than guessing it.

Return a JSON object with exactly these keys: {', '.join(FIELDS)}"""

# Xira rasmlarda model tug'ilgan sana qatorini "Berilgan sanasi" deb o'qib
# yuboradi. Yechim: undan tug'ilgan sanani ham SO'RASH - shunda unga qo'yadigan
# joy paydo bo'ladi va u issue_date ga tiqilmaydi. Sinovda bu xatoni yo'qotdi.
DATE_PROMPT = """This Uzbekistan ID card contains SEVERAL dates. Read the LABEL
next to each one, then report all three:

  issue_date  - labelled "Berilgan sanasi" / "Date of issue"
  expiry_date - labelled "Amal qilish muddati" / "Date of expiry",
                normally exactly 10 years after issue_date
  birth_date  - labelled "Tug'ilgan sanasi" / "Date of birth", decades earlier

Do not confuse them - match each date to its printed label, not to its position.
Use "" for any date you genuinely cannot read.

Return JSON: {"issue_date": "YYYY-MM-DD", "expiry_date": "YYYY-MM-DD", "birth_date": "YYYY-MM-DD"}"""

MRZ_PROMPT = """Look at the machine-readable zone (MRZ) at the bottom of this ID card:
3 monospaced lines, each exactly 30 characters, using < as filler.

Copy all 3 lines EXACTLY, character by character. Do not correct, reformat or
guess anything - digits and < signs matter. Return JSON: {"mrz": ["...", "...", "..."]}"""


# --- Model bilan aloqa ------------------------------------------------------

def ask_model(images: list[str], prompt: str, prefill: str = "{") -> dict:
    """Modelga so'rov yuboradi va JSON obyekt qaytaradi.

    `prefill` javob boshini belgilaydi - thinking'ni o'chirish uchun ham,
    modelni to'g'ridan-to'g'ri JSON yozishga majburlash uchun ham kerak.
    """
    # Prefill FAQAT Ollama yo'lida ma'noga ega (qwen3-vl thinking'ini
    # o'chiradi). vLLM'da assistant javobini oldindan to'ldirib bo'lmaydi —
    # u yerda JSON `response_format` bilan majburlanadi. Farqni `vision_chat`
    # o'zi hal qiladi va TO'LIQ matn qaytaradi.
    text = vision_chat(
        prompt,
        images,
        prefill=NO_THINK + prefill,
        json_mode=True,
        options={
            "temperature": 0,
            "num_ctx": config.NUM_CTX,
            "num_predict": config.NUM_PREDICT,
        },
        timeout=TIMEOUT,
    )
    if not config.USE_VLLM and text.startswith(NO_THINK):
        # Ollama yo'lida qaytgan matn boshida NO_THINK bloki turadi;
        # `_parse_json` ga faqat JSON qismi kerak.
        text = text[len(NO_THINK):]
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """Prefill tufayli javob yarim JSON bo'lishi mumkin - uni tiklaydi."""
    text = text.strip()
    for candidate in (text, text + "}", text + "\"}", text + "]}"):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Model JSON qaytarmadi:\n{text[:400]}")


# --- Normalizatsiya --------------------------------------------------------

DATE_RE = re.compile(r"(\d{1,4})[.\-/](\d{1,2})[.\-/](\d{2,4})")
PLACEHOLDERS = {"JOHN", "DOE", "JOHN DOE", "STRING", "N/A", "NULL", "NONE", "<KEY>", "XXX"}


def norm_date(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    m = DATE_RE.fullmatch(value)
    if not m:
        return None
    a, b, c = m.groups()
    if len(a) == 4:  # YYYY-M-D
        y, mo, d = a, b, c
    else:  # DD.MM.YYYY
        d, mo, y = a, b, c
    y = y if len(y) == 4 else ("20" + y if int(y) <= 30 else "19" + y)
    if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return None
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def norm_doc_number(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", value.upper())
    return cleaned or None


def norm_pinfl(value) -> str | None:
    if not isinstance(value, str):
        return None
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 14 else None


def norm_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" .,")
    # Model ba'zan MRZ bo'lagini yoki shablon qiymatni maydonga tiqib qo'yadi.
    if "<<" in text or text.upper() in PLACEHOLDERS:
        return None
    return text or None


def normalize(raw: dict) -> dict:
    return {
        "document_number": norm_doc_number(raw.get("document_number")),
        "pinfl": norm_pinfl(raw.get("pinfl")),
        "issue_date": norm_date(raw.get("issue_date")),
        "expiry_date": norm_date(raw.get("expiry_date")),
        "issued_by": norm_text(raw.get("issued_by")),
    }


def read_mrz(images: list[str]) -> dict | None:
    try:
        raw = ask_model(images, MRZ_PROMPT, prefill='{"mrz": [')
    except (RuntimeError, ValueError, requests.RequestException):
        return None
    lines = raw.get("mrz")
    if not isinstance(lines, list) or not lines:
        return None
    return mrz_lib.parse_td1("\n".join(str(x) for x in lines)).to_dict()


def cross_check(fields: dict, mrz_data: dict | None) -> dict:
    """MRZ checksumdan o'tgan bo'lsa u ishonchliroq - farq bo'lsa MRZ qiymati olinadi.

    Checksumdan o'tmagan MRZ e'tiborsiz qoldiriladi (aks holda to'g'ri vizual
    qiymatni xato MRZ qiymatiga almashtirib yuborish xavfi bor).
    """
    if not mrz_data or not mrz_data.get("valid"):
        return fields

    # Faqat MRZ da mavjud maydonlar. issue_date va issued_by MRZ da yo'q.
    for key in ("document_number", "pinfl", "expiry_date"):
        if mrz_data.get(key):
            fields[key] = mrz_data[key]
    return fields


def read_fields(img, extra=None) -> dict:
    """Tayyorlangan rasmdan maydonlarni bir marta o'qiydi."""
    images = [preprocess.to_b64(img)]
    if extra is not None:
        images.append(preprocess.to_b64(extra))
    return validate.repair(normalize(ask_model(images, PROMPT)))


def fix_dates(fields: dict, img) -> dict:
    """Sanalar shubhali bo'lsa, faqat sanalarga qaratilgan qayta so'rov.

    JSHSHIR ichidagi tug'ilgan sana - hakam: agar maqsadli o'qish o'sha
    sanani mustaqil ravishda to'g'ri topsa, demak model qatorlarni to'g'ri
    ajratgan va uning issue/expiry qiymatlari ishonchli.
    """
    try:
        raw = ask_model([preprocess.to_b64(img)], DATE_PROMPT)
    except (RuntimeError, ValueError):
        return fields

    focused = {"issue_date": norm_date(raw.get("issue_date")),
               "expiry_date": norm_date(raw.get("expiry_date"))}
    if not focused["issue_date"]:
        return fields

    pinfl_info = validate.parse_pinfl(fields.get("pinfl"))
    trusted = (pinfl_info is not None
               and norm_date(raw.get("birth_date")) == pinfl_info["birth_date"])

    candidate = validate.repair({**fields, **{k: v for k, v in focused.items() if v}})
    if trusted or validate.check(candidate)[0] > validate.check(fields)[0]:
        return candidate
    return fields


def find_orientation(img, extra=None) -> tuple:
    """To'g'ri burchakni topadi: har birini o'qib, ballari bo'yicha tanlaydi.

    Model teskari rasmdan ham "javob" qaytaraveradi, shuning uchun burchakni
    modeldan so'rash foydasiz - natijaning ichki mantiqqa mosligi bo'yicha
    hukm qilamiz (validate.check). To'liq ball olinsa qolganlari sinalmaydi.
    """
    best = (None, None, -1)  # (fields, img, score)
    for angle in preprocess.orientation_order(img):
        ready = preprocess.prepare(img, angle, level=1)
        ready_extra = preprocess.prepare(extra, angle, level=1) if extra is not None else None
        try:
            fields = read_fields(ready, ready_extra)
        except (RuntimeError, ValueError):
            continue
        score, _ = validate.check(fields)
        if score > best[2]:
            best = (fields, ready, score)
        if validate.is_confident(fields):
            break  # bu burchak aniq to'g'ri - qidirishni to'xtatamiz
    return best


def extract(front, back=None) -> dict:
    """Kartadan 5 ta maydonni o'qiydi.

    front/back - rasm fayl yo'li (Path) yoki rasm baytlari (bytes).

    Rasm qiyshiq, burilgan yoki xira bo'lishi mumkin, shuning uchun bir
    o'qish bilan cheklanmaymiz: burchak topiladi, natija ichki mantiqqa
    tekshiriladi va shubhali bo'lsa boshqa sifat darajalarida qayta o'qib
    ovoz berish orqali yakunlanadi. MRZ faqat tuzatish uchun ishlatiladi.
    """
    img = preprocess.load(front)
    extra = preprocess.load(back) if back is not None else None

    fields, ready, _ = find_orientation(img, extra)
    if fields is None:
        raise RuntimeError("Rasmni o'qib bo'lmadi")

    # Natija shubhali - boshqa sifat darajalarida qayta o'qib, ovoz beramiz.
    # Bitta raqamdagi OCR xatosi variantlar bo'ylab takrorlanmaydi.
    if not validate.is_confident(fields):
        candidates = [fields]
        for level in (2, 0):
            try:
                variant = preprocess.enhance(ready, level)
                candidates.append(read_fields(variant))
            except (RuntimeError, ValueError):
                continue
            if len(candidates) >= 2 and validate.is_confident(candidates[-1]):
                break
        merged = validate.repair(validate.merge(candidates))
        # Ovoz berish natijasi yomonlashtirmasligi kerak.
        if validate.check(merged)[0] >= validate.check(fields)[0]:
            fields = merged

    # Sanalar hali ham shubhali bo'lsa - maqsadli qayta so'rov.
    if any("date" in p for p in validate.check(fields)[1]):
        fields = fix_dates(fields, ready)

    if USE_MRZ:
        mrz_img = preprocess.prepare(extra if extra is not None else img,
                                     0, level=1) if extra is not None else ready
        mrz_data = read_mrz([preprocess.to_b64(preprocess.mrz_strip(mrz_img), quality=95)])
        if not mrz_data or not mrz_data["valid"]:
            retry = read_mrz([preprocess.to_b64(mrz_img)])
            if retry and (retry["valid"] or not mrz_data):
                mrz_data = retry
        fields = cross_check(fields, mrz_data)

    return validate.drop_impossible(fields)


def main() -> int:
    global MODEL, OLLAMA_URL

    p = argparse.ArgumentParser(description="O'zbekiston ID kartasidan ma'lumot olish")
    p.add_argument("front", type=Path, help="Kartaning old tomoni (rasm)")
    p.add_argument("--back", type=Path, help="Kartaning orqa tomoni (MRZ shu yerda)")
    p.add_argument("-o", "--output", type=Path, help="Natijani JSON faylga yozish")
    p.add_argument("--model", default=MODEL, help=f"Ollama modeli (default: {MODEL})")
    p.add_argument("--url", default=OLLAMA_URL, help="Ollama server manzili")
    args = p.parse_args()
    MODEL, OLLAMA_URL = args.model, args.url.rstrip("/")

    for path in (args.front, args.back):
        if path and not path.is_file():
            print(f"Fayl topilmadi: {path}", file=sys.stderr)
            return 1

    try:
        result = extract(args.front, args.back)
    except (RuntimeError, ValueError) as e:
        print(f"Xato: {e}", file=sys.stderr)
        return 1

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Yozildi: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
