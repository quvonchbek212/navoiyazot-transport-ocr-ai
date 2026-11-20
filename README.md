# Navoiyazot Logistics · Document OCR Service

Automated transport and identity document OCR recognition engine deployed at **Navoiyazot industrial plant logistics checkpoints and weighbridges**.

---

## Key Capabilities

* **Driver License (Prava) OCR**: Extracts driver full name, license series/number, category entitlements (B, C, D, E), and expiry dates.
* **Vehicle Technical Passport (Texpasport) OCR**: Reads license plate number, VIN code, chassis number, vehicle model, unladen weight, and gross vehicle mass (GVM).
* **National ID Card & Passport OCR**: Reads MRZ (Machine Readable Zone) lines, personal identification number (PINFL), and date of birth.
* **Image Preprocessing**: Auto-rotates, corrects skew, enhances contrast, and crops document boundaries before parsing.

---

## Tech Stack

* **Language**: Python 3.12+
* **Framework**: FastAPI (Multipart file upload)
* **Image Processing**: Pillow (PIL), NumPy
* **Extraction**: RegEx pattern matching, MRZ parser, validation checksums

---

## Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ocr/prava` | Upload image of driver's license and receive structured data |
| `POST` | `/ocr/texpasport` | Upload vehicle registration certificate (front/back) |
| `POST` | `/ocr/id-card` | Extract national biometric ID card fields and MRZ |
| `POST` | `/ocr/pasport` | Extract biometric passport fields and MRZ lines |
| `GET` | `/healthz` | Service health status |

---

## Quick Start

### 1. Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Service
```bash
python main.py
# Server runs on http://localhost:8007 (Docs: http://localhost:8007/docs)
```
