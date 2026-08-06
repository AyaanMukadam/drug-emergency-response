"""
Drug Emergency Response Prediction — Flask Backend
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
import re

app = Flask(__name__)

# ─── LOAD DATASET FOR BROWSER ─────────────────────────────────────────────────
_DATASET_CACHE = None

# If deployed (drug.csv not present locally), download from GitHub Release
DATASET_URL = os.environ.get(
    "DATASET_URL",
    "https://github.com/AyaanMukadam/drug-emergency-response/releases/download/v1.0/drug.csv.csv"
)

def ensure_dataset():
    """Download drug.csv if it doesn't exist (for cloud deployment)."""
    csv_path = "data/drug.csv"
    if os.path.exists(csv_path):
        return True
    print("[*] drug.csv not found locally — attempting download...")
    try:
        import urllib.request
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, csv_path)
        print(f"[OK] Dataset downloaded to {csv_path}")
        return True
    except Exception as e:
        print(f"[!] Dataset download failed: {e}")
        print("    Set DATASET_URL environment variable with a direct link to drug.csv")
        return False

def get_dataset():
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    ensure_dataset()
    try:
        df = pd.read_csv("data/drug.csv", on_bad_lines='skip')
        df.dropna(subset=['drugName', 'condition', 'rating'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').clip(1, 10)
        df.dropna(subset=['rating'], inplace=True)
        df['rating'] = df['rating'].astype(int)
        df['emergency'] = df['rating'].apply(lambda r:
            'CRITICAL' if r <= 3 else 'HIGH' if r <= 5 else 'MODERATE' if r <= 7 else 'LOW')
        df = df.reset_index(drop=True)
        df['id'] = df.index + 1
        _DATASET_CACHE = df
        print(f"[OK] Dataset cache loaded: {len(df):,} rows")
    except Exception as e:
        print(f"[!] Dataset load error: {e}")
        _DATASET_CACHE = pd.DataFrame()
    return _DATASET_CACHE


# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
MODEL_DIR = "model"

def load_models():
    models = {}
    try:
        models['vectorizer']         = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
        models['rating_model']       = joblib.load(f"{MODEL_DIR}/rating_model.pkl")
        models['sentiment_model']    = joblib.load(f"{MODEL_DIR}/sentiment_model.pkl")
        models['label_encoder']      = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        # v2: dedicated 4-class emergency classifier
        emerg_path = f"{MODEL_DIR}/emergency_model.pkl"
        if os.path.exists(emerg_path):
            models['emergency_model']   = joblib.load(emerg_path)
            models['emergency_encoder'] = joblib.load(f"{MODEL_DIR}/emergency_encoder.pkl")
        with open(f"{MODEL_DIR}/analytics.json") as f:
            models['analytics'] = json.load(f)
        models['drug_recs'] = pd.read_csv(f"{MODEL_DIR}/drug_recommendations.csv")
        with open(f"{MODEL_DIR}/metadata.json") as f:
            models['metadata'] = json.load(f)
        print("[OK] All models loaded successfully")
    except Exception as e:
        print(f"[!] Model loading error: {e}")
        print("    Run train_model.py first!")
    return models

MODELS = load_models()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&amp;|&quot;|&lt;|&gt;', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_combined(drug, condition, review):
    """Mirror the same feature engineering used at training time."""
    d = clean_text(drug)
    c = clean_text(condition)
    r = clean_text(review)
    return f"{d} {d} {d} {c} {c} {c} {r}"

def rating_to_emergency(rating):
    if rating <= 3:   return {"level": "CRITICAL", "color": "#ff4560", "icon": "\U0001f534", "desc": "Immediate medical attention recommended"}
    elif rating <= 5: return {"level": "HIGH",     "color": "#ff6b35", "icon": "\U0001f7e0", "desc": "High-risk response — monitor closely"}
    elif rating <= 7: return {"level": "MODERATE", "color": "#f59e0b", "icon": "\U0001f7e1", "desc": "Moderate risk — standard observation"}
    else:             return {"level": "LOW",      "color": "#00c896", "icon": "\U0001f7e2", "desc": "Low risk — positive drug response"}

def rating_to_emoji(rating):
    if rating <= 3:   return "\U0001f623"
    elif rating <= 5: return "\U0001f615"
    elif rating <= 7: return "\U0001f610"
    else:             return "\U0001f60a"

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    analytics = MODELS.get('analytics', {})
    metadata  = MODELS.get('metadata', {})
    return render_template('index.html', analytics=analytics, metadata=metadata)

@app.route('/predict')
def predict_page():
    conditions = MODELS.get('metadata', {}).get('conditions', [])
    drugs      = MODELS.get('metadata', {}).get('drugs', [])
    return render_template('predict.html', conditions=conditions, drugs=drugs)

@app.route('/analytics')
def analytics_page():
    analytics = MODELS.get('analytics', {})
    return render_template('analytics.html', analytics=analytics)

@app.route('/recommend')
def recommend_page():
    conditions = MODELS.get('metadata', {}).get('conditions', [])
    return render_template('recommend.html', conditions=conditions)

# ─── API ENDPOINTS ────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data   = request.get_json()
        review = data.get('review', '').strip()
        drug   = data.get('drug', 'Unknown').strip() or 'Unknown'
        cond   = data.get('condition', 'Unknown').strip() or 'Unknown'

        if len(review) < 10:
            return jsonify({"error": "Review must be at least 10 characters"}), 400

        if not MODELS.get('vectorizer'):
            return jsonify({"error": "Models not loaded. Run train_model.py first."}), 503

        # Build combined feature (same as training)
        combined = build_combined(drug, cond, review)
        vec      = MODELS['vectorizer'].transform([combined])

        # ── Rating prediction ────────────────────────────────────────────────
        raw_rating  = float(MODELS['rating_model'].predict(vec)[0])
        pred_rating = round(max(1.0, min(10.0, raw_rating)), 1)

        # ── Emergency level — prefer direct classifier if available ──────────
        if MODELS.get('emergency_model'):
            emerg_proba  = MODELS['emergency_model'].predict_proba(vec)[0]
            emerg_idx    = int(np.argmax(emerg_proba))
            emerg_label  = MODELS['emergency_encoder'].classes_[emerg_idx]
            emerg_conf   = round(float(emerg_proba[emerg_idx]) * 100, 1)
            # Build emergency dict using label
            label_map = {
                'CRITICAL': {"color": "#ff4560", "icon": "\U0001f534", "desc": "Immediate medical attention recommended"},
                'HIGH':     {"color": "#ff6b35", "icon": "\U0001f7e0", "desc": "High-risk response — monitor closely"},
                'MODERATE': {"color": "#f59e0b", "icon": "\U0001f7e1", "desc": "Moderate risk — standard observation"},
                'LOW':      {"color": "#00c896", "icon": "\U0001f7e2", "desc": "Low risk — positive drug response"},
            }
            emergency = {"level": emerg_label, **label_map.get(emerg_label, label_map['MODERATE'])}
            confidence = emerg_conf
            # Also align predicted rating with emergency level for display
            rating_ranges = {'CRITICAL': (1,3), 'HIGH': (4,5), 'MODERATE': (6,7), 'LOW': (8,10)}
            lo, hi = rating_ranges[emerg_label]
            # Blend: 60% direct prediction, 40% pull toward level midpoint
            level_mid   = (lo + hi) / 2
            pred_rating = round(max(lo, min(hi, 0.6*pred_rating + 0.4*level_mid)), 1)
        else:
            emergency  = rating_to_emergency(pred_rating)
            confidence = round(100 - abs(raw_rating - round(raw_rating)) * 20, 1)
            confidence = max(50, min(99, confidence))

        # ── Sentiment prediction ──────────────────────────────────────────────
        sent_proba = MODELS['sentiment_model'].predict_proba(vec)[0]
        sent_idx   = int(np.argmax(sent_proba))
        sentiment  = MODELS['label_encoder'].classes_[sent_idx]
        sent_conf  = round(float(sent_proba[sent_idx]) * 100, 1)

        emoji      = rating_to_emoji(pred_rating)
        word_count = len(review.split())

        return jsonify({
            "drug":       drug,
            "condition":  cond,
            "review_len": word_count,
            "rating": {
                "predicted":  pred_rating,
                "confidence": confidence,
                "display":    f"{pred_rating}/10",
                "emoji":      emoji
            },
            "sentiment": {
                "label":      sentiment,
                "confidence": sent_conf,
                "all_proba":  {
                    cls: round(float(p)*100, 1)
                    for cls, p in zip(MODELS['label_encoder'].classes_, sent_proba)
                }
            },
            "emergency": emergency
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    condition = request.args.get('condition', '').strip()
    if not condition:
        return jsonify({"error": "condition parameter required"}), 400

    df = MODELS.get('drug_recs')
    if df is None:
        return jsonify({"error": "Recommendation data not loaded"}), 503

    results = df[df['condition'].str.lower() == condition.lower()].head(10)
    if results.empty:
        # fuzzy match
        mask = df['condition'].str.lower().str.contains(condition.lower(), na=False)
        results = df[mask].head(10)

    if results.empty:
        return jsonify({"drugs": [], "message": "No matches found"})

    drugs_list = []
    for _, row in results.iterrows():
        rating = float(row['avg_rating'])
        drugs_list.append({
            "drugName":      row['drugName'],
            "avg_rating":    rating,
            "review_count":  int(row['review_count']),
            "emergency":     rating_to_emergency(rating)['level'],
            "emergency_color": rating_to_emergency(rating)['color'],
            "stars":         "★" * round(rating/2) + "☆" * (5 - round(rating/2))
        })

    return jsonify({"condition": condition, "drugs": drugs_list})


@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    return jsonify(MODELS.get('analytics', {}))



@app.route('/api/search_conditions', methods=['GET'])
def api_search_conditions():
    q = request.args.get('q', '').lower()
    conditions = MODELS.get('metadata', {}).get('conditions', [])
    matches = [c for c in conditions if q in c.lower()][:15]
    return jsonify(matches)


@app.route('/api/search_drugs', methods=['GET'])
def api_search_drugs():
    q = request.args.get('q', '').lower()
    drugs = MODELS.get('metadata', {}).get('drugs', [])
    matches = [d for d in drugs if q in d.lower()][:15]
    return jsonify(matches)


@app.route('/api/emergency_records', methods=['GET'])
def api_emergency_records():
    """Return dataset rows filtered by emergency level with pagination and search."""
    level   = request.args.get('level', '').upper().strip()
    page    = max(1, int(request.args.get('page', 1)))
    per_page = int(request.args.get('per_page', 50))
    search  = request.args.get('search', '').strip().lower()

    valid_levels = {'CRITICAL', 'HIGH', 'MODERATE', 'LOW'}
    if level not in valid_levels:
        return jsonify({"error": f"Invalid level. Use one of: {', '.join(valid_levels)}"}), 400

    df = get_dataset()
    if df.empty:
        return jsonify({"error": "Dataset not available"}), 503

    subset = df[df['emergency'] == level].copy()

    # Apply search filter across drugName and condition
    if search:
        mask = (
            subset['drugName'].str.lower().str.contains(search, na=False) |
            subset['condition'].str.lower().str.contains(search, na=False)
        )
        subset = subset[mask]

    total   = len(subset)
    start   = (page - 1) * per_page
    end     = start + per_page
    page_df = subset.iloc[start:end]

    emergency_colors = {
        'CRITICAL': '#ef4444', 'HIGH': '#f97316',
        'MODERATE': '#f59e0b', 'LOW':  '#10b981'
    }

    records = []
    for _, row in page_df.iterrows():
        records.append({
            "id":        int(row['id']),
            "drugName":  str(row['drugName']),
            "condition": str(row['condition']),
            "rating":    int(row['rating']),
            "emergency": level,
            "color":     emergency_colors[level],
        })

    return jsonify({
        "level":      level,
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "total_pages": max(1, -(-total // per_page)),  # ceiling div
        "records":    records
    })


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Drug Emergency Response Prediction")
    print("  http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

