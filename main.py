"""Navoiyazot Logistics: Transport & Driver Document OCR Service."""
import logging
import uvicorn
from fastapi import FastAPI
from ocr.id_card.router import router as id_card_router
from ocr.pasport.router import router as pasport_router
from ocr.prava.router import router as prava_router
from ocr.texpasport.router import router as texpasport_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Navoiyazot Logistics · Document OCR API",
    description="Automated text extraction for driver licenses, passports, ID cards, and vehicle registration certificates.",
    version="1.0.0"
)

app.include_router(id_card_router)
app.include_router(pasport_router)
app.include_router(prava_router)
app.include_router(texpasport_router)

@app.get("/healthz", tags=["Health"])
async def healthz():
    return {"status": "ok", "service": "ocr"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8007, reload=False)
