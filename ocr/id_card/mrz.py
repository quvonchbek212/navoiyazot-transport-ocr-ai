"""TD1 (ID-karta) MRZ zonasini tahlil qilish va check-digit tekshirish.

O'zbekiston ID kartasi orqa tomonida 3 qator x 30 belgidan iborat TD1 MRZ bor:

  IDUZB<hujjat_raqami><cd><ixtiyoriy: JSHSHIR>
  <tug'ilgan><cd><jins><amal_muddati><cd>UZB<ixtiyoriy><umumiy_cd>
  FAMILIYA<<ISM<OTASINING<ISMI<<<<<<<
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

WEIGHTS = (7, 3, 1)
FILLER = "<"


def _char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if ch == FILLER:
        return 0
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    raise ValueError(f"MRZ uchun yaroqsiz belgi: {ch!r}")


def check_digit(data: str) -> int:
    """ICAO 9303 bo'yicha 7-3-1 nazorat raqamini hisoblaydi."""
    return sum(_char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(data)) % 10


def _verify(data: str, digit: str) -> bool:
    if not digit.isdigit():
        return False
    try:
        return check_digit(data) == int(digit)
    except ValueError:
        return False


def _date(yymmdd: str, *, future_window: bool) -> str | None:
    """YYMMDD -> YYYY-MM-DD. `future_window` amal muddati uchun (kelajakka moyil)."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    # TD1 da faqat 2 xonali yil bor: tug'ilgan sana o'tmishda, muddat kelajakda.
    century = 2000 if (future_window or yy <= 30) else 1900
    return f"{century + yy:04d}-{mm:02d}-{dd:02d}"


def _clean(name: str) -> str:
    return " ".join(p for p in name.replace(FILLER, " ").split() if p)


@dataclass
class MrzResult:
    valid: bool
    document_type: str | None = None
    issuing_country: str | None = None
    document_number: str | None = None
    pinfl: str | None = None
    birth_date: str | None = None
    sex: str | None = None
    expiry_date: str | None = None
    nationality: str | None = None
    surname: str | None = None
    given_names: str | None = None
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_lines(raw: str) -> list[str]:
    """Model qaytargan matndan 3 ta 30-belgili MRZ qatorini ajratib oladi."""
    lines = []
    for line in raw.replace("«", "<").replace("＜", "<").upper().splitlines():
        line = re.sub(r"[^A-Z0-9<]", "", line)
        if len(line) >= 28:
            lines.append(line[:30].ljust(30, FILLER))
    return lines[:3]


def parse_td1(raw: str) -> MrzResult:
    lines = normalize_lines(raw)
    if len(lines) != 3:
        return MrzResult(valid=False, errors=[f"3 ta MRZ qatori topilmadi (topildi: {len(lines)})"])

    l1, l2, l3 = lines
    doc_number_raw, doc_cd = l1[5:14], l1[14]
    optional1 = l1[15:30]
    birth_raw, birth_cd = l2[0:6], l2[6]
    sex_raw = l2[7]
    expiry_raw, expiry_cd = l2[8:14], l2[14]
    nationality = l2[15:18].replace(FILLER, "")
    optional2 = l2[18:29]
    composite_cd = l2[29]

    composite_src = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    checks = {
        "document_number": _verify(doc_number_raw, doc_cd),
        "birth_date": _verify(birth_raw, birth_cd),
        "expiry_date": _verify(expiry_raw, expiry_cd),
        "composite": _verify(composite_src, composite_cd),
    }

    surname, _, given = l3.partition("<<")
    pinfl = re.sub(r"[^0-9]", "", optional1) or re.sub(r"[^0-9]", "", optional2)

    result = MrzResult(
        valid=all(checks.values()),
        document_type=l1[0:2].replace(FILLER, ""),
        issuing_country=l1[2:5].replace(FILLER, ""),
        document_number=doc_number_raw.replace(FILLER, ""),
        pinfl=pinfl if len(pinfl) == 14 else None,
        birth_date=_date(birth_raw, future_window=False),
        sex={"M": "M", "F": "F"}.get(sex_raw),
        expiry_date=_date(expiry_raw, future_window=True),
        nationality=nationality,
        surname=_clean(surname),
        given_names=_clean(given),
        checks=checks,
    )
    result.errors = [f"{k} nazorat raqami mos kelmadi" for k, ok in checks.items() if not ok]
    return result
