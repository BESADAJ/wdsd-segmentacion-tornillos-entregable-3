# Imagen reproducible del dashboard: solo inferencia en CPU (ONNX Runtime). La usa Render y sirve para correrlo local:
#   docker build -t wdsd-sgw .  &&  docker run -p 8050:8050 wdsd-sgw   ->  http://localhost:8050
FROM python:3.11.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8050
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py MODEL_CARD.md ./
COPY pages/ pages/
COPY src/__init__.py src/data.py src/plots.py src/inference.py src/explain.py src/ui.py src/
COPY assets/ assets/
COPY data/ data/
COPY results/confirmatorio/ results/confirmatorio/
COPY artifacts/ artifacts/
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8050
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 180
