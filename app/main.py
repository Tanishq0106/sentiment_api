from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import time
import datetime
import os

from app.predict import predict_sentiment, predict_batch

# ── App initialisation ────────────────────────────────────────
app = FastAPI(
    title       = "Sentiment Analysis API",
    description = "SVM-based movie review sentiment classifier",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ── CORS middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# ── Serve frontend static files ───────────────────────────────
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

# ── In-memory metrics store ───────────────────────────────────
metrics = {
    "total_requests"    : 0,
    "positive_count"    : 0,
    "negative_count"    : 0,
    "total_latency_ms"  : 0.0,
    "request_log"       : [],       # last 20 requests
    "startup_time"      : datetime.datetime.utcnow().isoformat()
}

# ── Pydantic schemas ──────────────────────────────────────────
class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length = 10,
        max_length = 10000,
        description = "The movie review text to classify"
    )
    include_cleaned: Optional[bool] = Field(
        False,
        description = "Whether to return the cleaned/preprocessed text"
    )

    @field_validator('review')
    @classmethod
    def review_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Review text cannot be blank or whitespace only")
        return v.strip()


class BatchRequest(BaseModel):
    reviews: list[str] = Field(
        ...,
        min_length = 1,
        max_length = 50,
        description = "List of reviews to classify (max 50)"
    )

    @field_validator('reviews')
    @classmethod
    def validate_each_review(cls, reviews):
        for i, r in enumerate(reviews):
            if not r.strip():
                raise ValueError(f"Review at index {i} is empty")
            if len(r) < 5:
                raise ValueError(f"Review at index {i} is too short (min 5 chars)")
        return [r.strip() for r in reviews]


class SentimentResponse(BaseModel):
    label           : str
    confidence      : float
    confidence_pct  : str
    raw_length      : int
    clean_length    : int
    cleaned_text    : Optional[str] = None
    latency_ms      : float


class BatchResponse(BaseModel):
    results         : list[dict]
    total           : int
    positive_count  : int
    negative_count  : int
    latency_ms      : float


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the web app frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Used by Docker, Render, and load balancers to verify the service is alive.
    """
    return {
        "status"      : "healthy",
        "model"       : "SVM Linear (C=1.0)",
        "version"     : "1.0.0",
        "timestamp"   : datetime.datetime.utcnow().isoformat(),
        "uptime_since": metrics["startup_time"]
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: ReviewRequest):
    """
    Classify a single movie review as positive or negative.
    Returns label, confidence score, token counts, and latency.
    """
    t0 = time.time()

    try:
        result = predict_sentiment(request.review)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms = round((time.time() - t0) * 1000, 2)

    # ── Update metrics ────────────────────────────────────────
    metrics["total_requests"]   += 1
    metrics["total_latency_ms"] += latency_ms
    if result["label"] == "positive":
        metrics["positive_count"] += 1
    else:
        metrics["negative_count"] += 1

    # Keep a rolling log of last 20 requests
    metrics["request_log"].append({
        "timestamp" : datetime.datetime.utcnow().isoformat(),
        "label"     : result["label"],
        "confidence": result["confidence"],
        "latency_ms": latency_ms
    })
    if len(metrics["request_log"]) > 20:
        metrics["request_log"].pop(0)

    # ── Build response ────────────────────────────────────────
    response = SentimentResponse(
        label          = result["label"],
        confidence     = result["confidence"],
        confidence_pct = f"{result['confidence']*100:.2f}%",
        raw_length     = result["raw_length"],
        clean_length   = result["clean_length"],
        latency_ms     = latency_ms,
        cleaned_text   = result["cleaned_text"] if request.include_cleaned else None
    )
    return response


@app.post("/batch", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """
    Classify up to 50 reviews in a single request.
    More efficient than calling /predict repeatedly.
    """
    t0 = time.time()

    try:
        results = predict_batch(request.reviews)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    latency_ms    = round((time.time() - t0) * 1000, 2)
    pos_count     = sum(1 for r in results if r["label"] == "positive")
    neg_count     = len(results) - pos_count

    metrics["total_requests"]   += len(results)
    metrics["positive_count"]   += pos_count
    metrics["negative_count"]   += neg_count
    metrics["total_latency_ms"] += latency_ms

    return BatchResponse(
        results        = results,
        total          = len(results),
        positive_count = pos_count,
        negative_count = neg_count,
        latency_ms     = latency_ms
    )


@app.get("/metrics")
async def get_metrics():
    """
    Returns live usage statistics.
    Shows total requests, sentiment distribution, avg latency, recent log.
    """
    total = metrics["total_requests"]
    avg_latency = (
        round(metrics["total_latency_ms"] / total, 2)
        if total > 0 else 0
    )
    return {
        "total_requests"    : total,
        "positive_count"    : metrics["positive_count"],
        "negative_count"    : metrics["negative_count"],
        "positive_rate"     : f"{(metrics['positive_count']/total*100):.1f}%" if total > 0 else "N/A",
        "avg_latency_ms"    : avg_latency,
        "recent_requests"   : metrics["request_log"][-10:],
        "uptime_since"      : metrics["startup_time"]
    }