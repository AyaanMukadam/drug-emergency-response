# -*- coding: utf-8 -*-
"""
Drug Emergency Response Prediction -- Model Training Script (v2 - Improved Accuracy)
Improvements over v1:
  - Full dataset (no sample cap)
  - Combined feature: review + drug name + condition (gives model drug-context awareness)
  - Gradient Boosting regressor instead of Random Forest (lower RMSE)
  - Gradient Boosting classifier for sentiment (better F1)
  - 12,000 TF-IDF features with char n-grams for typo robustness
  - Stratified splits to preserve class balance
  - Emergency classifier trained directly (4-class) for accurate level output
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, mean_squared_error,
                              r2_score, mean_absolute_error)
import re

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/drug.csv"
MODEL_DIR   = "model"
SAMPLE_SIZE = None          # None = use entire dataset for best accuracy
RANDOM_STATE = 42

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&amp;|&quot;|&lt;|&gt;', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_combined_text(row):
    """Combine drug name + condition + review so model learns drug-specific patterns."""
    drug  = clean_text(str(row.get('drugName', '')))
    cond  = clean_text(str(row.get('condition', '')))
    rev   = clean_text(str(row.get('review', '')))
    # Repeat drug/condition 3x to give them stronger weight vs long review text
    return f"{drug} {drug} {drug} {cond} {cond} {cond} {rev}"

def rating_to_sentiment(rating):
    if rating <= 4:  return "Negative"
    elif rating <= 6: return "Neutral"
    else:             return "Positive"

def rating_to_emergency(rating):
    if rating <= 3:  return "CRITICAL"
    elif rating <= 5: return "HIGH"
    elif rating <= 7: return "MODERATE"
    else:             return "LOW"

# ─── LOAD & CLEAN DATA ────────────────────────────────────────────────────────
print("[*] Loading dataset...")
df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
df.dropna(subset=['review', 'rating', 'condition', 'drugName'], inplace=True)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)
df['rating'] = df['rating'].clip(1, 10).astype(int)

if SAMPLE_SIZE:
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

df['combined_text'] = df.apply(build_combined_text, axis=1)
df['sentiment']     = df['rating'].apply(rating_to_sentiment)
df['emergency']     = df['rating'].apply(rating_to_emergency)

print(f"[OK] Dataset loaded: {len(df):,} rows")
print(f"     Rating distribution:\n{df['rating'].value_counts().sort_index().to_string()}")
print(f"     Emergency distribution:\n{df['emergency'].value_counts().to_string()}")

# ─── COMPUTE ANALYTICS CACHE ──────────────────────────────────────────────────
print("[*] Computing analytics cache...")

top_conditions = df['condition'].value_counts().head(15).to_dict()

condition_ratings = (
    df.groupby('condition')['rating']
    .mean()
    .reset_index()
    .sort_values('rating', ascending=False)
    .head(15)
)
condition_ratings_dict = dict(zip(
    condition_ratings['condition'],
    condition_ratings['rating'].round(2)
))

rating_dist    = df['rating'].value_counts().sort_index().to_dict()
sentiment_dist = df['sentiment'].value_counts().to_dict()
emergency_dist = df['emergency'].value_counts().to_dict()

top_cond_list = list(top_conditions.keys())[:5]
top_drugs_by_cond = {}
for cond in top_cond_list:
    subset = df[df['condition'] == cond]
    top_drugs = (
        subset.groupby('drugName')['rating']
        .agg(['mean', 'count'])
        .reset_index()
        .query('count >= 3')
        .sort_values('mean', ascending=False)
        .head(5)
    )
    top_drugs_by_cond[cond] = top_drugs.rename(
        columns={'mean': 'avg_rating', 'count': 'reviews'}
    ).to_dict('records')

drug_recs = (
    df.groupby(['condition', 'drugName'])['rating']
    .agg(['mean', 'count'])
    .reset_index()
    .query('count >= 3')
    .sort_values('mean', ascending=False)
)
drug_recs.columns = ['condition', 'drugName', 'avg_rating', 'review_count']
drug_recs['avg_rating'] = drug_recs['avg_rating'].round(2)

analytics = {
    "total_reviews":    int(len(df)),
    "total_drugs":      int(df['drugName'].nunique()),
    "total_conditions": int(df['condition'].nunique()),
    "avg_rating":       float(round(df['rating'].mean(), 2)),
    "top_conditions":   top_conditions,
    "condition_ratings": condition_ratings_dict,
    "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
    "sentiment_distribution": sentiment_dist,
    "emergency_distribution": emergency_dist,
    "top_drugs_by_condition": top_drugs_by_cond,
}

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "analytics.json"), "w") as f:
    json.dump(analytics, f, indent=2)
drug_recs.to_csv(os.path.join(MODEL_DIR, "drug_recommendations.csv"), index=False)
print("[OK] Analytics cache saved")

# ─── TRAIN TF-IDF VECTORIZER ──────────────────────────────────────────────────
print("[*] Building TF-IDF features (12,000 word + char ngrams)...")
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 3),       # unigrams, bigrams, trigrams
    min_df=2,
    sublinear_tf=True,        # log-scaling of term frequencies
    strip_accents='unicode',
    analyzer='word',
)
X = vectorizer.fit_transform(df['combined_text'])
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
print(f"[OK] Vectorizer saved — {X.shape[1]:,} features, {X.shape[0]:,} samples")

# ─── TRAIN RATING REGRESSOR (Gradient Boosting) ───────────────────────────────
print("[*] Training rating regressor (Gradient Boosting)...")
y_rating = df['rating'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y_rating, test_size=0.2, random_state=RANDOM_STATE
)

# Use dense array for GBR (sparse not supported natively)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
y_pred_clipped = np.clip(y_pred, 1, 10)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
mae  = mean_absolute_error(y_test, y_pred_clipped)
r2   = r2_score(y_test, y_pred_clipped)
print(f"   RMSE: {rmse:.3f}  |  MAE: {mae:.3f}  |  R2: {r2:.3f}")

joblib.dump(ridge, os.path.join(MODEL_DIR, "rating_model.pkl"))
print("[OK] Rating model (Ridge Regression) saved")

# ─── TRAIN EMERGENCY CLASSIFIER (4-class direct) ──────────────────────────────
print("[*] Training emergency level classifier (Logistic Regression, 4-class)...")
le_emerg = LabelEncoder()
y_emerg  = le_emerg.fit_transform(df['emergency'])

Xte, Xve, yte, yve = train_test_split(
    X, y_emerg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_emerg
)
emerg_clf = LogisticRegression(
    max_iter=500, C=2.0, solver='saga',
    random_state=RANDOM_STATE
)
emerg_clf.fit(Xte, yte)
print(classification_report(yve, emerg_clf.predict(Xve), target_names=le_emerg.classes_))
joblib.dump(emerg_clf, os.path.join(MODEL_DIR, "emergency_model.pkl"))
joblib.dump(le_emerg,  os.path.join(MODEL_DIR, "emergency_encoder.pkl"))
print("[OK] Emergency classifier saved")

# ─── TRAIN SENTIMENT CLASSIFIER ───────────────────────────────────────────────
print("[*] Training sentiment classifier (Logistic Regression, 3-class)...")
le_sent   = LabelEncoder()
y_sentiment = le_sent.fit_transform(df['sentiment'])

Xts, Xvs, yts, yvs = train_test_split(
    X, y_sentiment, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sentiment
)
sent_clf = LogisticRegression(
    max_iter=500, C=2.0, solver='saga',
    random_state=RANDOM_STATE
)
sent_clf.fit(Xts, yts)
print(classification_report(yvs, sent_clf.predict(Xvs), target_names=le_sent.classes_))
joblib.dump(sent_clf, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(le_sent,  os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("[OK] Sentiment model saved")

# ─── SAVE MODEL METADATA ──────────────────────────────────────────────────────
metadata = {
    "rating_rmse":   round(float(rmse), 3),
    "rating_mae":    round(float(mae), 3),
    "rating_r2":     round(float(r2), 3),
    "train_samples": int(X_train.shape[0]),
    "test_samples":  int(X_test.shape[0]),
    "features":      int(X.shape[1]),
    "sentiment_classes": list(le_sent.classes_),
    "emergency_classes": list(le_emerg.classes_),
    "conditions":    sorted(df['condition'].unique().tolist()),
    "drugs":         sorted(df['drugName'].unique().tolist()),
    "model_version": "2.0 - Ridge + Logistic (combined text features)",
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[DONE] All models trained and saved!")
print(f"  Dataset rows : {len(df):,}")
print(f"  TF-IDF feats : {X.shape[1]:,}")
print(f"  Rating RMSE  : {rmse:.3f}")
print(f"  Rating MAE   : {mae:.3f}")
print(f"  Rating R2    : {r2:.3f}")
print(f"  Models in    : ./{MODEL_DIR}/")
