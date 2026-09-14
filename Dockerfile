FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY prompts/ ./prompts/
COPY static/ ./static/

# Render กำหนดพอร์ตให้ผ่าน environment variable PORT แบบไดนามิก
# ต้อง bind กับ 0.0.0.0:$PORT เท่านั้น ห้าม hardcode พอร์ต 8000
# (ถ้ารันเองโดยไม่ตั้ง PORT จะ fallback ไปที่ 8000 ให้ทดสอบบนเครื่องได้ตามปกติ)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
