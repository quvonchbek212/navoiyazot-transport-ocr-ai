"""Rasmni model uchun tayyorlash: burilish, qiyshiqlik, xiralik.

Model rasmni o'zi to'g'rilay olmaydi - teskari yoki xira rasmda u taxmin
qilishga o'tadi. Shuning uchun rasm modelga berilishidan oldin tuzatiladi.
"""

from __future__ import annotations

import io
import math

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

# ID-1 standarti: 85.6 x 54 mm. Karta har doim gorizontal (landscape).
CARD_RATIO = 85.6 / 54  # ~1.585
MIN_WORK_WIDTH = 1000   # bundan kichik rasm oldin kattalashtiriladi
MAX_SIDE = 1600


def load(src) -> Image.Image:
    """Fayl yo'li yoki baytlardan rasm ochadi, EXIF burchagini qo'llaydi."""
    handle = io.BytesIO(src) if isinstance(src, bytes) else src
    return ImageOps.exif_transpose(Image.open(handle)).convert("RGB")


def orientation_order(img: Image.Image) -> list[int]:
    """Sinab ko'rish uchun burchaklar ro'yxati - eng ehtimoli birinchi.

    Karta gorizontal bo'lgani uchun tik rasm deyarli aniq 90 yoki 270 ga
    burilgan. Bu bilan birinchi urinishdayoq to'g'ri kelish ehtimoli oshadi.
    """
    if img.width >= img.height:
        return [0, 180, 90, 270]
    return [90, 270, 0, 180]


def _projection_variance(binary: Image.Image, angle: float) -> float:
    """Rasmni burib, gorizontal proyeksiya profilining dispersiyasini o'lchaydi.

    Matn qatorlari gorizontal bo'lganda har bir qatorda yo faqat matn, yo
    faqat bo'shliq bo'ladi - qator yig'indilari keskin farqlanadi va
    dispersiya maksimal bo'ladi. Qiyshiq bo'lsa qatorlar aralashib, profil
    tekislanadi.
    """
    rotated = binary.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    width, height = rotated.size
    pixels = rotated.load()
    rows = [sum(pixels[x, y] for x in range(0, width, 2))
            for y in range(0, height)]
    mean = sum(rows) / len(rows)
    return sum((r - mean) ** 2 for r in rows) / len(rows)


def deskew(img: Image.Image, max_angle: float = 12.0) -> Image.Image:
    """Kichik qiyshiqlikni (skew) to'g'rilaydi."""
    # Chekkalardagi fon/hoshiyalar o'lchovni buzmasligi uchun markazni olamiz.
    gray = ImageOps.autocontrast(img.convert("L"))
    w, h = gray.size
    gray = gray.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    gray = gray.resize((360, max(60, round(360 * gray.height / gray.width))))

    # Matnni ikkilik (binary) holga keltiramiz: qorong'i piksel = matn = 1.
    threshold = ImageStat.Stat(gray).mean[0] * 0.75
    binary = gray.point(lambda p: 1 if p < threshold else 0, mode="1")

    best_angle, best_score = 0.0, _projection_variance(binary, 0.0)
    for step in range(1, int(max_angle) + 1):
        for angle in (-step, step):
            score = _projection_variance(binary, angle)
            if score > best_score:
                best_angle, best_score = float(angle), score

    if abs(best_angle) < 1.0:  # arzimas qiyshiqlik - tegmaymiz
        return img
    return img.rotate(best_angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255))


def blur_score(img: Image.Image) -> float:
    """Rasm qanchalik aniq (yuqori = aniqroq). Chekka energiyasining dispersiyasi."""
    edges = img.convert("L").resize((320, 200)).filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).stddev[0] ** 2


def enhance(img: Image.Image, level: int = 1) -> Image.Image:
    """Xira rasmni o'qishga yaroqli holga keltiradi.

    level 0 - faqat kontrast (toza rasmlar uchun)
    level 1 - kontrast + o'tkirlashtirish (standart)
    level 2 - kulrang + kuchli o'tkirlashtirish (juda xira rasmlar uchun)
    """
    # Mayda matn (JSHSHIR, karta raqami) uchun yetarli o'lcham kerak.
    if img.width < MIN_WORK_WIDTH:
        scale = MIN_WORK_WIDTH / img.width
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)

    if level >= 2:
        img = ImageOps.grayscale(img).convert("RGB")

    img = ImageOps.autocontrast(img, cutoff=1)

    if level >= 1:
        percent = 150 if level == 1 else 220
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))
    if level >= 2:
        img = ImageEnhance.Contrast(img).enhance(1.35)

    return img


def prepare(img: Image.Image, angle: int = 0, level: int = 1,
            straighten: bool = True) -> Image.Image:
    """To'liq tayyorlash: burish -> qiyshiqlikni to'g'rilash -> sifatni oshirish."""
    if angle:
        img = img.rotate(-angle, expand=True)  # soat yo'nalishida
    if straighten:
        img = deskew(img)
    img = enhance(img, level)
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    return img


def mrz_strip(img: Image.Image) -> Image.Image:
    """MRZ pastki qismini kesib, kattalashtiradi."""
    strip = img.crop((0, int(img.height * 0.60), img.width, img.height))
    target = min(2 * strip.width, 2000)
    if target > strip.width:
        scale = target / strip.width
        strip = strip.resize((round(strip.width * scale), round(strip.height * scale)),
                             Image.LANCZOS)
    return strip


def to_b64(img: Image.Image, quality: int = 92) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def looks_like_card(img: Image.Image) -> bool:
    """Nisbat ID-1 kartaga o'xshaydimi (kesilgan rasmlarni aniqlash uchun)."""
    ratio = max(img.size) / min(img.size)
    return math.isclose(ratio, CARD_RATIO, rel_tol=0.25)
