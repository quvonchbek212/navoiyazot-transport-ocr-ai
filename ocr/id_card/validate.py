"""O'qilgan maydonlarning ichki mantiqqa mosligini tekshirish.

Bu modul modelsiz ishlaydi va ikki vazifani bajaradi:
  1. Bir necha variantdan (burchak, sifat) qaysi biri to'g'ri o'qilganini
     tanlash - ballari eng yuqorisi g'olib.
  2. Xato o'qilgan maydonni aniqlab, uni qayta so'rashga asos berish.

JSHSHIR (PINFL) tuzilishi shu yerda asosiy quroldir: uning ichida tug'ilgan
sana va jins kodlangan, shuning uchun bitta raqam xato o'qilsa ham ko'pincha
sana yaroqsiz bo'lib qoladi va bu aniqlanadi.
"""

from __future__ import annotations

import re
from datetime import date

DOC_RE = re.compile(r"^[A-Z]{2}\d{7}$")
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Kartaning amal qilish muddati: 5 yoki 10 yil (kunlarda, kichik zaxira bilan).
VALIDITY_DAYS = (5 * 365, 10 * 365)
VALIDITY_TOLERANCE = 60

# O'zbekistonda ID-karta 16 yoshdan beriladi.
MIN_ISSUE_AGE_YEARS = 16


def _as_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    m = DATE_RE.match(value)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def parse_pinfl(pinfl) -> dict | None:
    """JSHSHIR ichidagi tug'ilgan sana va jinsni ochadi.

    Tuzilishi: [asr/jins][DD][MM][YY][ketma-ket raqamlar]
      1,2 - 1800-yillar   3,4 - 1900-yillar   5,6 - 2000-yillar
      toq  - erkak        juft - ayol
    """
    if not isinstance(pinfl, str) or not re.fullmatch(r"\d{14}", pinfl):
        return None
    lead = int(pinfl[0])
    if not 1 <= lead <= 6:
        return None
    century = {1: 1800, 2: 1800, 3: 1900, 4: 1900, 5: 2000, 6: 2000}[lead]
    try:
        birth = date(century + int(pinfl[5:7]), int(pinfl[3:5]), int(pinfl[1:3]))
    except ValueError:
        return None
    if birth > date.today():
        return None
    return {"birth_date": birth.isoformat(), "sex": "M" if lead % 2 else "F"}


def check(fields: dict) -> tuple[int, list[str]]:
    """Maydonlarni baholaydi: (ball, muammolar ro'yxati).

    Ball qanchalik yuqori bo'lsa, o'qish shunchalik ishonchli. Turli
    variantlarni solishtirish uchun ishlatiladi.
    """
    score, problems = 0, []

    doc = fields.get("document_number")
    if doc and DOC_RE.match(doc):
        score += 3
    else:
        problems.append("document_number")

    pinfl_info = parse_pinfl(fields.get("pinfl"))
    if pinfl_info:
        score += 3
    else:
        problems.append("pinfl")

    issue = _as_date(fields.get("issue_date"))
    expiry = _as_date(fields.get("expiry_date"))

    if issue and issue <= date.today():
        score += 1
    else:
        problems.append("issue_date")

    if expiry:
        score += 1
    else:
        problems.append("expiry_date")

    # Sanalar bir-biriga mos kelishi kerak - eng kuchli tekshiruvlardan biri.
    if issue and expiry:
        if expiry <= issue:
            problems.append("issue_date/expiry_date tartibi teskari")
        else:
            gap = (expiry - issue).days
            if any(abs(gap - v) <= VALIDITY_TOLERANCE for v in VALIDITY_DAYS):
                score += 3
            else:
                problems.append(f"amal muddati g'ayrioddiy ({gap // 365} yil)")

    # JSHSHIR ichidagi tug'ilgan sana - mustaqil nazorat nuqtasi.
    # ID-karta 16 yoshdan beriladi, shuning uchun berilgan sana tug'ilgan
    # sanadan kamida shuncha keyin bo'lishi shart. Xira rasmda model tug'ilgan
    # sana qatorini issue_date deb o'qib yuboradi - bu qoida shuni ushlaydi.
    if pinfl_info and issue:
        birth = _as_date(pinfl_info["birth_date"])
        age_days = (issue - birth).days
        if age_days < MIN_ISSUE_AGE_YEARS * 365:
            problems.append("issue_date tug'ilgan sanaga juda yaqin "
                            "(tug'ilgan sana qatori o'qilgan bo'lishi mumkin)")
            score -= 3
        elif (issue.month, issue.day) == (birth.day, birth.month):
            # Kun/oy almashib ketgani - tug'ilgan sana qatoridan olingan belgisi.
            problems.append("issue_date tug'ilgan sana bilan kun/oy jihatdan mos")
            score -= 1

    if fields.get("issued_by"):
        score += 1

    return score, problems


MAX_SCORE = 12


def is_confident(fields: dict) -> bool:
    """Barcha asosiy tekshiruvlardan o'tdimi - qayta urinish shart emasmi."""
    score, problems = check(fields)
    return not problems and score >= MAX_SCORE - 1


def repair(fields: dict) -> dict:
    """Aniq va xavfsiz tuzatishlar (taxminga asoslanmagan)."""
    issue = _as_date(fields.get("issue_date"))
    expiry = _as_date(fields.get("expiry_date"))
    # Model bu ikkisini almashtirib qo'yishi mumkin - tartib aniq noto'g'ri
    # bo'lsa va almashtirilganda to'g'ri chiqsa, joyini almashtiramiz.
    if issue and expiry and expiry < issue:
        gap = (issue - expiry).days
        if any(abs(gap - v) <= VALIDITY_TOLERANCE for v in VALIDITY_DAYS):
            fields["issue_date"], fields["expiry_date"] = (
                fields["expiry_date"], fields["issue_date"])
    return fields


def drop_impossible(fields: dict) -> dict:
    """Mantiqan bo'lishi mumkin bo'lmagan qiymatni null qiladi.

    Xato qiymatni qaytargandan ko'ra "o'qiy olmadim" degani ma'qul: xato
    sana tashqi ko'rinishdan to'g'ridek tuyuladi va sezilmay o'tib ketadi.
    """
    issue = _as_date(fields.get("issue_date"))
    expiry = _as_date(fields.get("expiry_date"))
    if issue and expiry and expiry <= issue:
        fields["expiry_date"] = None
    return fields


def merge(candidates: list[dict]) -> dict:
    """Bir necha o'qishdan har bir maydon uchun eng ko'p takrorlangan qiymat.

    Bitta raqamdagi OCR xatosi odatda takrorlanmaydi, shuning uchun ovoz
    berish aynan shunday xatolarni yaxshi filtrlaydi.
    """
    if not candidates:
        return {}
    result = {}
    for key in candidates[0]:
        votes: dict[str, int] = {}
        for cand in candidates:
            value = cand.get(key)
            if value:
                votes[value] = votes.get(value, 0) + 1
        result[key] = max(votes, key=votes.get) if votes else None
    return result
