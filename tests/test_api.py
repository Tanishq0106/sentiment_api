import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health check ──────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


# ── Single prediction — positive ──────────────────────────────
def test_predict_positive():
    response = client.post("/predict", json={
        "review": "This film was an absolute masterpiece. "
                  "Beautifully acted and directed with stunning visuals."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["confidence"] > 0.7
    assert "latency_ms" in data
    assert "confidence_pct" in data


# ── Single prediction — negative ──────────────────────────────
def test_predict_negative():
    response = client.post("/predict", json={
        "review": "Terrible waste of time. The worst film I have ever seen. "
                  "Awful acting and a nonsensical plot."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "negative"
    assert data["confidence"] > 0.7


# ── Validation — too short ────────────────────────────────────
def test_predict_too_short():
    response = client.post("/predict", json={"review": "Bad"})
    assert response.status_code == 422


# ── Validation — empty string ─────────────────────────────────
def test_predict_empty():
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 422


# ── Validation — missing field ────────────────────────────────
def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ── Batch prediction ──────────────────────────────────────────
def test_batch_predict():
    response = client.post("/batch", json={
        "reviews": [
            "Absolutely brilliant film with outstanding performances.",
            "Complete garbage, boring and poorly written.",
            "One of the greatest movies ever made, a true classic."
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["positive_count"] + data["negative_count"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["label"] == "positive"
    assert data["results"][1]["label"] == "negative"


# ── Batch — exceeds limit ─────────────────────────────────────
def test_batch_exceeds_limit():
    reviews = ["This is a test review that is long enough." for _ in range(51)]
    response = client.post("/batch", json={"reviews": reviews})
    assert response.status_code == 422


# ── Metrics endpoint ──────────────────────────────────────────
def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "positive_count" in data
    assert "avg_latency_ms" in data


# ── Frontend serves HTML ──────────────────────────────────────
def test_frontend_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Sentiment" in response.text


# ── include_cleaned flag ──────────────────────────────────────
def test_predict_with_cleaned_text():
    response = client.post("/predict", json={
        "review" : "A wonderfully crafted film with superb acting throughout.",
        "include_cleaned": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["cleaned_text"] is not None
    assert len(data["cleaned_text"]) > 0