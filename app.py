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


@app.route('/api/analytics_detail', methods=['GET'])
def api_analytics_detail():
    """
    Drill-down analytics for click events on the Analytics page.
    ?type=sentiment&value=Positive
    ?type=emergency&value=CRITICAL
    ?type=rating&value=8
    ?type=condition&value=Depression
    ?type=stat&value=conditions|drugs|reviews|avg_rating
    """
    detail_type = request.args.get('type', '').lower().strip()
    value       = request.args.get('value', '').strip()

    if not detail_type or not value:
        return jsonify({'error': 'type and value parameters required'}), 400

    # ── STAT CARD DRILL-DOWN (uses analytics.json only — no CSV needed) ──────
    if detail_type == 'stat':
        analytics = MODELS.get('analytics', {})
        if value == 'conditions':
            top_conds = analytics.get('top_conditions', {})
            cond_ratings = analytics.get('condition_ratings', {})
            items = [
                {'condition': k, 'reviews': v, 'avg_rating': cond_ratings.get(k, 0)}
                for k, v in top_conds.items()
            ]
            return jsonify({'type': 'stat', 'value': 'conditions', 'color': '#00b4d8',
                            'title': 'Top Conditions by Review Volume', 'items': items})
        elif value == 'drugs':
            top_drugs_by_cond = analytics.get('top_drugs_by_condition', {})
            all_drugs = []
            for cond, drugs in top_drugs_by_cond.items():
                for d in drugs[:2]:
                    all_drugs.append({
                        'drugName': d['drugName'],
                        'condition': cond,
                        'avg_rating': round(d['avg_rating'], 2),
                        'reviews': d.get('reviews', 0)
                    })
            all_drugs.sort(key=lambda x: x['avg_rating'], reverse=True)
            return jsonify({'type': 'stat', 'value': 'drugs', 'color': '#a855f7',
                            'title': 'Top-Rated Drugs (by Condition)', 'items': all_drugs[:15]})
        elif value == 'reviews':
            rating_dist = analytics.get('rating_distribution', {})
            sent_dist   = analytics.get('sentiment_distribution', {})
            return jsonify({
                'type': 'stat', 'value': 'reviews', 'color': '#1a6bff',
                'title': 'Review Overview',
                'total': analytics.get('total_reviews', 0),
                'rating_distribution': rating_dist,
                'sentiment_distribution': sent_dist,
            })
        elif value == 'avg_rating':
            emerg_dist   = analytics.get('emergency_distribution', {})
            sent_dist    = analytics.get('sentiment_distribution', {})
            return jsonify({
                'type': 'stat', 'value': 'avg_rating', 'color': '#00c896',
                'title': 'Overall Rating Insights',
                'avg_rating': analytics.get('avg_rating', 0),
                'emergency_distribution': emerg_dist,
                'sentiment_distribution': sent_dist,
            })
        else:
            return jsonify({'error': f'Unknown stat value: {value}'}), 400

    # For other types, we need the live dataset
    df = get_dataset()
    if df.empty:
        return jsonify({'error': 'Dataset not available. Please ensure data/drug.csv is present.'}), 503


    emergency_colors = {
        'CRITICAL': '#ff4560', 'HIGH': '#ff6b35',
        'MODERATE': '#f59e0b', 'LOW':  '#00c896'
    }
    sentiment_colors = {
        'Positive': '#00d4aa', 'Negative': '#ff4560', 'Neutral': '#f59e0b'
    }

    try:
        # ── SENTIMENT DRILL-DOWN ─────────────────────────────────────────────
        if detail_type == 'sentiment':
            # Map rating -> sentiment (using same label encoder logic from training)
            label_map = {'Positive': (8, 10), 'Negative': (1, 5), 'Neutral': (6, 7)}
            if value in label_map:
                lo, hi = label_map[value]
                subset = df[(df['rating'] >= lo) & (df['rating'] <= hi)]
            else:
                # Try using sentiment_distribution key
                subset = df[df['rating'] >= 8] if value == 'Positive' else df[df['rating'] <= 5]

            top_conds = (
                subset.groupby('condition').size()
                .sort_values(ascending=False).head(10)
                .reset_index(name='count')
            )
            top_drugs = (
                subset.groupby('drugName')['rating']
                .agg(['mean','count']).reset_index()
                .rename(columns={'mean':'avg_rating','count':'reviews'})
                .sort_values('avg_rating', ascending=False).head(8)
            )
            emerg_dist = subset['emergency'].value_counts().to_dict()

            return jsonify({
                'type':    'sentiment',
                'value':   value,
                'color':   sentiment_colors.get(value, '#00d4aa'),
                'total':   int(len(subset)),
                'avg_rating': round(float(subset['rating'].mean()), 2),
                'top_conditions': [
                    {'condition': r['condition'], 'count': int(r['count'])}
                    for _, r in top_conds.iterrows()
                ],
                'top_drugs': [
                    {'drugName': str(r['drugName']),
                     'avg_rating': round(float(r['avg_rating']), 2),
                     'reviews': int(r['reviews'])}
                    for _, r in top_drugs.iterrows()
                ],
                'emergency_distribution': {
                    k: int(v) for k, v in emerg_dist.items()
                },
            })

        # ── EMERGENCY DRILL-DOWN ─────────────────────────────────────────────
        elif detail_type == 'emergency':
            level = value.upper()
            subset = df[df['emergency'] == level]
            top_conds = (
                subset.groupby('condition').size()
                .sort_values(ascending=False).head(10)
                .reset_index(name='count')
            )
            top_drugs = (
                subset.groupby('drugName')['rating']
                .agg(['mean','count']).reset_index()
                .rename(columns={'mean':'avg_rating','count':'reviews'})
                .sort_values('reviews', ascending=False).head(8)
            )
            sent_dist = {}
            for rating, row in subset['rating'].value_counts().sort_index().items():
                sent_dist[str(rating)] = int(row)

            descriptions = {
                'CRITICAL': 'Immediate medical attention recommended',
                'HIGH':     'High-risk response — monitor closely',
                'MODERATE': 'Moderate risk — standard observation',
                'LOW':      'Low risk — positive drug response',
            }

            return jsonify({
                'type':       'emergency',
                'value':      level,
                'color':      emergency_colors.get(level, '#00d4aa'),
                'total':      int(len(subset)),
                'avg_rating': round(float(subset['rating'].mean()), 2),
                'description': descriptions.get(level, ''),
                'top_conditions': [
                    {'condition': r['condition'], 'count': int(r['count'])}
                    for _, r in top_conds.iterrows()
                ],
                'top_drugs': [
                    {'drugName': str(r['drugName']),
                     'avg_rating': round(float(r['avg_rating']), 2),
                     'reviews': int(r['reviews'])}
                    for _, r in top_drugs.iterrows()
                ],
                'rating_distribution': sent_dist,
            })

        # ── RATING DRILL-DOWN ────────────────────────────────────────────────
        elif detail_type == 'rating':
            try:
                rating_val = int(value)
            except ValueError:
                return jsonify({'error': 'value must be an integer 1-10'}), 400
            subset = df[df['rating'] == rating_val]
            top_conds = (
                subset.groupby('condition').size()
                .sort_values(ascending=False).head(10)
                .reset_index(name='count')
            )
            top_drugs = (
                subset.groupby('drugName').size()
                .sort_values(ascending=False).head(8)
                .reset_index(name='count')
            )
            emergency_level = (
                'CRITICAL' if rating_val <= 3 else
                'HIGH'     if rating_val <= 5 else
                'MODERATE' if rating_val <= 7 else 'LOW'
            )
            color = (
                '#ff4560' if rating_val <= 3 else
                '#ff6b35' if rating_val <= 5 else
                '#f59e0b' if rating_val <= 7 else '#00c896'
            )
            return jsonify({
                'type':    'rating',
                'value':   str(rating_val),
                'color':   color,
                'total':   int(len(subset)),
                'emergency_level': emergency_level,
                'top_conditions': [
                    {'condition': r['condition'], 'count': int(r['count'])}
                    for _, r in top_conds.iterrows()
                ],
                'top_drugs': [
                    {'drugName': str(r['drugName']), 'reviews': int(r['count'])}
                    for _, r in top_drugs.iterrows()
                ],
            })

        # ── CONDITION DRILL-DOWN ─────────────────────────────────────────────
        elif detail_type == 'condition':
            subset = df[df['condition'].str.lower() == value.lower()]
            if subset.empty:
                mask = df['condition'].str.lower().str.contains(value.lower(), na=False)
                subset = df[mask]

            if subset.empty:
                return jsonify({'error': f'Condition "{value}" not found'}), 404

            top_drugs = (
                subset.groupby('drugName')['rating']
                .agg(['mean','count']).reset_index()
                .rename(columns={'mean':'avg_rating','count':'reviews'})
                .sort_values('avg_rating', ascending=False).head(8)
            )
            emerg_dist = subset['emergency'].value_counts().to_dict()
            rating_dist = {str(r): int(c) for r, c in subset['rating'].value_counts().sort_index().items()}

            return jsonify({
                'type':       'condition',
                'value':      value,
                'color':      '#00b4d8',
                'total':      int(len(subset)),
                'avg_rating': round(float(subset['rating'].mean()), 2),
                'top_drugs': [
                    {'drugName': str(r['drugName']),
                     'avg_rating': round(float(r['avg_rating']), 2),
                     'reviews': int(r['reviews'])}
                    for _, r in top_drugs.iterrows()
                ],
                'emergency_distribution': {k: int(v) for k, v in emerg_dist.items()},
                'rating_distribution': rating_dist,
            })

        # ── STAT CARD DRILL-DOWN ─────────────────────────────────────────────
        elif detail_type == 'stat':
            analytics = MODELS.get('analytics', {})
            if value == 'conditions':
                top_conds = analytics.get('top_conditions', {})
                cond_ratings = analytics.get('condition_ratings', {})
                items = [
                    {'condition': k, 'reviews': v, 'avg_rating': cond_ratings.get(k, 0)}
                    for k, v in top_conds.items()
                ]
                return jsonify({'type': 'stat', 'value': 'conditions', 'color': '#00b4d8',
                                'title': 'Top Conditions by Review Volume', 'items': items})
            elif value == 'drugs':
                top_drugs_by_cond = analytics.get('top_drugs_by_condition', {})
                all_drugs = []
                for cond, drugs in top_drugs_by_cond.items():
                    for d in drugs[:2]:
                        all_drugs.append({
                            'drugName': d['drugName'],
                            'condition': cond,
                            'avg_rating': round(d['avg_rating'], 2),
                            'reviews': d.get('reviews', 0)
                        })
                all_drugs.sort(key=lambda x: x['avg_rating'], reverse=True)
                return jsonify({'type': 'stat', 'value': 'drugs', 'color': '#a855f7',
                                'title': 'Top-Rated Drugs (by Condition)', 'items': all_drugs[:15]})
            elif value == 'reviews':
                rating_dist = analytics.get('rating_distribution', {})
                sent_dist   = analytics.get('sentiment_distribution', {})
                return jsonify({
                    'type': 'stat', 'value': 'reviews', 'color': '#1a6bff',
                    'title': 'Review Overview',
                    'total': analytics.get('total_reviews', 0),
                    'rating_distribution': rating_dist,
                    'sentiment_distribution': sent_dist,
                })
            elif value == 'avg_rating':
                emerg_dist   = analytics.get('emergency_distribution', {})
                sent_dist    = analytics.get('sentiment_distribution', {})
                return jsonify({
                    'type': 'stat', 'value': 'avg_rating', 'color': '#00c896',
                    'title': 'Overall Rating Insights',
                    'avg_rating': analytics.get('avg_rating', 0),
                    'emergency_distribution': emerg_dist,
                    'sentiment_distribution': sent_dist,
                })
            else:
                return jsonify({'error': f'Unknown stat value: {value}'}), 400

        else:
            return jsonify({'error': f'Unknown type: {detail_type}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

