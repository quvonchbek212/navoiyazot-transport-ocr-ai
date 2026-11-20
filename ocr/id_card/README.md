# O'zbekiston ID kartasi — ma'lumot ajratib olish

Ollama'dagi `qwen3-vl:8b` vision modeli orqali ID kartadagi barcha maydonlarni
o'qiydi va MRZ checksumlari bilan tekshiradi.

## REST API (Swagger)

Bu modul ATIL AI ning umumiy ilovasiga router bo'lib ulanadi — o'z serveri yo'q.
Loyiha ildizidan:

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs (bitta — barcha xizmatlar uchun umumiy).

Endpoint:

- `POST /id-card/extract` — `multipart/form-data`: `front` (majburiy), `back` (ixtiyoriy, MRZ shu yerda)

```bash
curl -F "front=@old.jpg" -F "back=@orqa.jpg" http://127.0.0.1:8000/id-card/extract
```

Xato kodlari: `415` noto'g'ri format, `413` 10 MB dan katta, `422` fayl yuborilmagan,
`502` Ollama'ga ulanib bo'lmadi, `504` model javob bermadi.

So'rov **10–60 soniya** (ba'zan ko'proq) davom etadi — ikkita model chaqiruvi bor.

Sozlash: loyiha ildizidagi `.env` — `OLLAMA_BASE_URL`, `OCR_OLLAMA_MODEL`, `OCR_USE_MRZ`.

## CLI orqali ishlatish

Loyiha ildizidan:

```bash
python -m ocr.id_card.extractor old.jpg                      # faqat old tomon
python -m ocr.id_card.extractor old.jpg --back orqa.jpg      # MRZ orqa tomonda
python -m ocr.id_card.extractor old.jpg -o natija.json
python -m ocr.id_card.extractor old.jpg --model qwen3-vl:30b # boshqa model
```

Chiqish kodi: `0` — MRZ bilan tasdiqlangan, `2` — o'qildi lekin tasdiqlanmadi, `1` — xato.

## Javob

Faqat 5 ta maydon qaytadi, o'qilmagani `null`:

```json
{
  "document_number": "AD8718879",
  "pinfl": "40305986080012",
  "issue_date": "2024-09-21",
  "expiry_date": "2034-09-20",
  "issued_by": "IIV 18224"
}
```

## Qiyin rasmlar bilan ishlash

Rasm burilgan, qiyshiq yoki xira bo'lishi mumkin. Quvur buni bosqichma-bosqich
hal qiladi ([preprocess.py](preprocess.py), [validate.py](validate.py)):

1. **Burchak.** Karta gorizontal (85.6×54 mm), shuning uchun tik rasm deyarli
   aniq 90/270 ga burilgan — shu tartibda sinaladi. Har bir burchakda o'qib,
   natija ichki mantiqqa tekshiriladi va ballari eng yuqorisi tanlanadi.
   Modeldan burchakni so'rash foydasiz: u teskari rasmdan ham "javob" beraveradi.
2. **Qiyshiqlik.** Proyeksiya profili usuli: rasm turli burchaklarga burilib,
   matn qatorlari eng keskin ajralgan burchak topiladi (±12°).
3. **Xiralik.** Kichik rasm kattalashtiriladi, kontrast va o'tkirlik oshiriladi.
4. **Shubhali natija.** Turli sifat darajalarida qayta o'qilib, har bir maydon
   uchun ovoz beriladi — bitta raqamdagi OCR xatosi variantlar bo'ylab
   takrorlanmaydi.
5. **Sanalar.** Ular hali ham shubhali bo'lsa, faqat sanalarga qaratilgan
   qayta so'rov yuboriladi.

### Maydonlarni adashtirmaslik

Model ko'p adashadigan joy — **tug'ilgan sana qatorini `issue_date` deb o'qish**.
Ikki himoya bor:

- Maqsadli so'rovda modeldan `birth_date` ham so'raladi. Shunda unga
  qo'yadigan joy paydo bo'ladi va u `issue_date` ga tiqilmaydi. Sinovda bu
  xatoni butunlay yo'qotdi.
- JSHSHIR ichida tug'ilgan sana kodlangan (`4|030598|...` → 03.05.1998).
  ID-karta 16 yoshdan berilgani uchun `issue_date` undan kamida shuncha keyin
  bo'lishi shart — aks holda natija past ball oladi va qayta o'qiladi.

### Sinov natijasi

Bir xil karta 6 xil holatda (`USE_MRZ` yoqilgan va o'chirilgan — ikkalasida ham):

| Holat | Natija | Vaqt |
|---|---|---|
| toza | 5/5 | ~5s |
| 90° burilgan | 5/5 | ~8s |
| 180° teskari | 5/5 | ~8s |
| 7° qiyshiq | 5/5 | ~6s |
| xira + kichik | 5/5 | ~5s |
| burilgan+qiyshiq+xira+yarim o'lcham | 5/5 | ~8s |

## MRZ nima qiladi

MRZ (orqa tomondagi 3 qatorli TD1 zonasi) ICAO 9303 nazorat raqamlariga ega.
`mrz.py` ularni qayta hisoblaydi. MRZ checksumdan **to'liq o'tsa**,
`document_number`, `pinfl`, `expiry_date` uning qiymatlari bilan jimgina
tuzatiladi. Checksumdan o'tmasa — butunlay e'tiborsiz qoldiriladi, chunki
xato o'qilgan MRZ to'g'ri vizual qiymatni buzib yuborishi mumkin.

Bu ichki mexanizm — javobga chiqmaydi.

`.env` da `OCR_USE_MRZ=false` qilinsa MRZ o'qilmaydi: javob ~2 barobar tez,
lekin tuzatish yo'q.

## Ishonchlilik haqida

Natija OCR modelining o'qishi. **Javobning o'zidan qiymat to'g'ri yoki
noto'g'ri ekanini bilib bo'lmaydi.** Sinovda `pinfl` da bitta raqam xato
o'qilgan holat uchradi (`312...` → `313...`) — tashqi ko'rinishdan buni
sezib bo'lmaydi. `issue_date` va `issued_by` MRZ da umuman yo'q, shuning
uchun ular hech qanday usul bilan tekshirilmaydi.

## Bu modelda aniqlangan muhim jihatlar

Quyidagilar sinov orqali topilgan va kodda hisobga olingan:

1. **`format` (JSON schema) rejimidan foydalanilmaydi.** Bu model bilan
   grammatika cheklovi generatsiyani 300 soniyadan oshirib yuboradi (ayniqsa
   `["string","null"]` union tiplarida), ba'zan esa model rasmni o'qish
   o'rniga `"John Doe"`, `"<KEY>"` kabi shablon qiymat qaytaradi.
   Buning o'rniga prefill + JSON tahlili ishlatilgan (~3 soniya).

2. **Thinking'ni o'chirish kerak.** `qwen3-vl` erkin qo'yilsa 8000+ belgi
   "o'ylab", javobga umuman yetib bormaydi (`done_reason: length`, bo'sh
   javob). `think: false` ham, `/no_think` ham bu versiyada ishlamadi.
   Yagona yechim — assistant javobini yopiq `<think></think>` bloki bilan
   oldindan to'ldirish (`NO_THINK`).

3. **ngrok bepul tunneli so'rovni aynan 300 soniyada uzadi** va HTML xato
   sahifasi qaytaradi. Timeout 280s qilib qo'yilgan, HTML javob esa aniq
   xato sifatida ushlanadi.

4. **MRZ verbatim o'qish ishonchsiz.** Sinovda `qwen3-vl:8b` ham,
   `qwen3-vl:30b` ham MRZ raqamlarini xato ko'chirdi (checksum o'tmadi).
   Shuning uchun MRZ pastki bo'lakni kesib, 2x kattalashtirib o'qiladi va
   bir marta qayta urinib ko'riladi. Baribir o'tmasa — natija
   `verified: false` bo'lib, ogohlantirish beriladi. **Yaroqsiz MRZ hech
   qachon jimgina to'g'ri deb qabul qilinmaydi.**

Amaliy natija: vizual maydonlar (ism, sana, karta raqami) yaxshi o'qiladi,
lekin `pinfl` va `issue_date` kabi maydonlarda xatolar uchraydi. Yuridik
ahamiyatga ega ishlarda natijani odam tekshirishi shart.

## Testlar

Modelsiz ishlaydi — internet yoki Ollama kerak emas. Loyiha ildizidan:

```bash
pytest tests -q                            # barchasi
pytest tests/test_id_card_mrz.py -q        # MRZ tahlili va nazorat raqamlari
pytest tests/test_id_card_validate.py -q   # maydonlarni baholash mantiqi
pytest tests/test_id_card_preprocess.py -q # burchak, qiyshiqlik, xiralik
```
