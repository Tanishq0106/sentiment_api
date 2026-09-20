# ── Stage 1: Base image ───────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="Tanishq"
LABEL description="SVM Sentiment Analysis API — FastAPI + scikit-learn"
LABEL version="1.0.0"

# ── Prevents Python from writing .pyc files to disk ───────────
ENV PYTHONDONTWRITEBYTECODE=1

# ── Prevents Python from buffering stdout/stderr ──────────────
# Without this, print() statements don't appear in Docker logs
ENV PYTHONUNBUFFERED=1

# ── Set working directory inside the container ────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────
# gcc is needed to compile some Python packages from source
# && chaining minimises Docker layers (each RUN = one layer)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker layer cache trick) ────────
# If requirements.txt hasn't changed, Docker reuses the cached
# pip install layer — massively speeds up rebuilds
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data inside the image ───────────────────────
# Must be done at build time so the container doesn't need
# internet access at runtime
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('wordnet', quiet=True)"

# ── Copy application code ─────────────────────────────────────
# Done AFTER pip install to maximise cache hits
# Changing app code doesn't re-run pip install
COPY app/        ./app/
COPY frontend/   ./frontend/
COPY artifacts/  ./artifacts/

# ── Expose port ───────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
# Docker checks this every 30s. If it fails 3 times,
# the container is marked unhealthy and restarted
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start command ─────────────────────────────────────────────
# host=0.0.0.0 binds to all interfaces (required inside Docker)
# workers=2 handles concurrent requests
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]