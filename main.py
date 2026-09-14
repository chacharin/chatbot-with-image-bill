"""
Chatbot องค์กร: ถามตอบทั่วไป + อ่านใบเสร็จเพื่อบันทึกรายจ่าย
FastAPI + OpenRouter — ดูรายละเอียดสถาปัตยกรรมใน plan.md (โฟลเดอร์แม่ของ app/)
"""

import base64
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

# ---------------------------------------------------------------------------
# การตั้งค่าและ path
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPTS_DIR = BASE_DIR / "prompts"
CHAT_PROMPT_PATH = PROMPTS_DIR / "chat_system.txt"
RECEIPT_PROMPT_PATH = PROMPTS_DIR / "receipt_system.txt"

STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "expenses.csv"

CSV_COLUMNS = ["saved_at", "payment_date", "receipt_no", "payee", "amount", "source_image"]

ALLOWED_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Receipt Chatbot")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Utility: prompts
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> str:
    """อ่านไฟล์ prompt ใหม่ทุกครั้งที่เรียก เพื่อให้แก้ไฟล์แล้วเห็นผลทันทีโดยไม่ต้อง restart"""
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"ไม่พบไฟล์ prompt: {path.name}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Utility: CSV
# ---------------------------------------------------------------------------

def ensure_csv() -> None:
    """สร้างไฟล์ CSV พร้อมหัวคอลัมน์ถ้ายังไม่มี เขียน BOM ครั้งเดียวตอนสร้างไฟล์เท่านั้น
    เพื่อไม่ให้เกิด BOM ซ้ำซ้อนทุกครั้งที่ append (ปัญหาที่พบบ่อยของ utf-8-sig + append mode)"""
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_expense_row(row: list) -> None:
    ensure_csv()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_expenses_sorted_desc() -> list[dict]:
    ensure_csv()
    with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # saved_at อยู่ในรูปแบบ YYYY-MM-DD HH:MM:SS.ffffff จึงเรียงแบบ string ได้ตรงกับเวลาจริง
    rows.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
    return rows


def get_expenses_context() -> Optional[str]:
    """สร้างข้อความสรุปรายจ่ายจาก CSV ไว้แนบเข้า context ของโหมดถามตอบทั่วไป
    ถ้าไฟล์ CSV ยังไม่มีอยู่จริง (ยังไม่เคยมีการบันทึกใบเสร็จเลย) จะไม่สร้างไฟล์ขึ้นมาเปล่า ๆ
    และคืนค่า None ทันที เพื่อไม่ให้ส่งข้อมูลว่างไปให้โมเดลโดยไม่จำเป็น"""
    if not CSV_PATH.exists():
        return None

    rows = read_expenses_sorted_desc()
    if not rows:
        return None

    lines = ["ข้อมูลรายจ่ายที่บันทึกไว้ในระบบ (เรียงจากรายการล่าสุดไปเก่าสุด):"]
    for i, r in enumerate(rows, 1):
        payment_date = r.get("payment_date") or "(ไม่ระบุ)"
        receipt_no = r.get("receipt_no") or "(ไม่ระบุ)"
        payee = r.get("payee") or "(ไม่ระบุ)"
        amount = r.get("amount") or "(ไม่ระบุ)"
        lines.append(
            f"{i}. จ่ายเมื่อ {payment_date} | เลขที่ใบเสร็จ {receipt_no} | "
            f"ผู้รับเงิน {payee} | จำนวนเงิน {amount} บาท"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility: normalize ข้อมูลที่โมเดลตอบกลับมา
# ---------------------------------------------------------------------------

def parse_receipt_json(raw: str) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_date(value) -> Optional[str]:
    """แปลง พ.ศ. เป็น ค.ศ. (ถ้าเลขปี > 2400) และคืนค่ารูปแบบ DD/MM/YYYY
    ถ้ารูปแบบไม่ตรงกับที่คาดไว้ จะคืนค่าดิบที่โมเดลตอบมาแทนการทิ้งข้อมูล"""
    if value is None or not isinstance(value, str):
        return None
    v = value.strip()
    if v.lower() in ("null", "none", ""):
        return None

    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", v)
    if not m:
        return v  # รูปแบบไม่คาดคิด เก็บค่าดิบไว้ดีกว่าทิ้งข้อมูล

    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    if year > 2400:
        year -= 543

    try:
        date(year, month, day)  # แค่ตรวจว่าเป็นวันที่จริงได้หรือไม่
    except ValueError:
        return v  # แปลงแล้วไม่ใช่วันที่จริง เก็บค่าดิบไว้

    return f"{day:02d}/{month:02d}/{year:04d}"


def normalize_amount(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("null", "none", ""):
            return None
        v = v.replace(",", "").replace("บาท", "").replace("฿", "").strip()
        try:
            return round(float(v), 2)
        except ValueError:
            return None
    return None


def clean_text_field(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("null", "none", ""):
            return None
        return v
    return str(value)


# ---------------------------------------------------------------------------
# Utility: เรียก OpenRouter
# ---------------------------------------------------------------------------

def require_openrouter_config() -> None:
    if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
        raise HTTPException(
            status_code=500,
            detail="ยังไม่ได้ตั้งค่า OPENROUTER_API_KEY หรือ OPENROUTER_MODEL ใน .env "
                   "(ดูตัวอย่างใน .env.example)",
        )


def call_openrouter(messages: list) -> str:
    require_openrouter_config()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages}
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"เรียก OpenRouter ไม่สำเร็จ: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter ตอบกลับผิดพลาด ({resp.status_code}): {resp.text[:500]}",
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"รูปแบบผลลัพธ์จาก OpenRouter ไม่ถูกต้อง: {data}") from exc


# ---------------------------------------------------------------------------
# Routes: หน้าเว็บ
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Routes: API
# ---------------------------------------------------------------------------

@app.get("/api/expenses")
def get_expenses():
    return JSONResponse(content=read_expenses_sorted_desc())


@app.post("/api/chat")
async def chat(message: str = Form(""), image: Optional[UploadFile] = File(None)):
    has_image = image is not None and bool(image.filename)

    if not has_image:
        # -------- โหมดถามตอบทั่วไป --------
        if not message.strip():
            raise HTTPException(status_code=400, detail="กรุณาพิมพ์ข้อความหรือแนบรูปใบเสร็จ")
        system_prompt = load_prompt(CHAT_PROMPT_PATH)
        chat_messages = [{"role": "system", "content": system_prompt}]

        expenses_context = get_expenses_context()
        if expenses_context:
            chat_messages.append({"role": "system", "content": expenses_context})

        chat_messages.append({"role": "user", "content": message})
        reply = call_openrouter(chat_messages)
        return {"type": "chat", "reply": reply}

    # -------- โหมดอ่านใบเสร็จ --------
    filename_lower = image.filename.lower()
    ext = Path(filename_lower).suffix
    mime = (image.content_type or "").lower()

    valid_ext = ext in ALLOWED_EXTENSIONS
    valid_mime = mime in ALLOWED_IMAGE_EXT_BY_MIME
    if not (valid_ext or valid_mime):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์รูปภาพ .jpg และ .png เท่านั้น")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="ไฟล์ภาพว่างเปล่าหรืออ่านไม่ได้")

    # ใช้ mime เป็นหลักในการตัดสินนามสกุลไฟล์ที่บันทึก ถ้าไม่รู้จัก mime ให้ fallback ไปตามนามสกุลเดิม
    save_ext = ALLOWED_IMAGE_EXT_BY_MIME.get(mime)
    if not save_ext:
        save_ext = ".png" if ext == ".png" else ".jpg"
    save_mime = "image/png" if save_ext == ".png" else "image/jpeg"

    now = datetime.now()
    saved_at_csv = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    source_image = now.strftime("%Y%m%d-%H%M%S-%f") + save_ext

    image_path = UPLOADS_DIR / source_image
    image_path.write_bytes(image_bytes)

    data_url = f"data:{save_mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    system_prompt = load_prompt(RECEIPT_PROMPT_PATH)
    user_text = message.strip() or "กรุณาอ่านข้อมูลจากใบเสร็จนี้"

    vision_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    raw = call_openrouter(vision_messages)
    parsed = parse_receipt_json(raw)

    if parsed is None:
        # ลองใหม่อีกหนึ่งครั้งด้วยคำสั่งย้ำให้ตอบเป็น JSON เท่านั้น
        retry_messages = vision_messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "กรุณาตอบใหม่เป็น JSON object เดียวเท่านั้น ห้ามมีข้อความอื่นล้อมรอบ"},
        ]
        raw_retry = call_openrouter(retry_messages)
        parsed = parse_receipt_json(raw_retry)

    if parsed is None:
        return JSONResponse(
            status_code=200,
            content={
                "type": "receipt_error",
                "reply": "ไม่สามารถอ่านข้อมูลจากใบเสร็จเป็น JSON ได้ กรุณาลองใหม่หรือถ่ายภาพให้ชัดขึ้น "
                         f"(บันทึกภาพต้นฉบับไว้ที่ uploads/{source_image} แล้ว แต่ยังไม่บันทึกลง CSV)",
                "raw_model_output": raw,
            },
        )

    payment_date = normalize_date(parsed.get("payment_date"))
    receipt_no = clean_text_field(parsed.get("receipt_no"))
    payee = clean_text_field(parsed.get("payee"))
    amount = normalize_amount(parsed.get("amount"))

    append_expense_row([
        saved_at_csv,
        payment_date or "",
        receipt_no or "",
        payee or "",
        amount if amount is not None else "",
        source_image,
    ])

    result = {
        "saved_at": saved_at_csv,
        "payment_date": payment_date,
        "receipt_no": receipt_no,
        "payee": payee,
        "amount": amount,
        "source_image": source_image,
    }

    return {"type": "receipt", "reply": "บันทึกรายจ่ายเรียบร้อยแล้ว", "data": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
