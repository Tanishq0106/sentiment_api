import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources if not already present
nltk.download('stopwords',   quiet=True)
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('wordnet',     quiet=True)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, 'artifacts', 'svm_best_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'artifacts', 'tfidf_vectorizer.pkl')

# ── Lazy model loader ─────────────────────────────────────────
_model      = None
_vectorizer = None

def get_model():
    """Load model once and cache it. Subsequent calls return cached version."""
    global _model, _vectorizer
    if _model is None:
        print("Loading model artifacts...")
        _model      = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model loaded successfully.")
    return _model, _vectorizer


def predict_sentiment(review: str) -> dict:
    model, vectorizer = get_model()

    cleaned    = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label      = "positive" if prediction == 1 else "negative"
    confidence = float(probability[prediction])

    return {
        "label"       : label,
        "confidence"  : round(confidence, 4),
        "cleaned_text": cleaned,
        "raw_length"  : len(review.split()),
        "clean_length": len(cleaned.split())
    }


def predict_batch(reviews: list[str]) -> list[dict]:
    model, vectorizer = get_model()

    cleaned       = [clean_text(r) for r in reviews]
    vectorized    = vectorizer.transform(cleaned)
    predictions   = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label      = "positive" if pred == 1 else "negative"
        confidence = float(prob[pred])
        results.append({
            "review_index": i,
            "label"       : label,
            "confidence"  : round(confidence, 4),
            "clean_length": len(cleaned[i].split())
        })
    return results

# ── Preprocessing (identical to training pipeline) ────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Applies the exact same cleaning pipeline used during training.
    Must be identical — any difference causes a train/serve skew bug.
    """
    text = text.lower()
    text = re.sub(r'<.*?>',        ' ', text)   # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text) # URLs
    text = re.sub(r'[^a-z\s]',     ' ', text)   # non-alpha
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)

