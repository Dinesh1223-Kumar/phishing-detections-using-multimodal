import pandas as pd
import joblib
import os
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from features.html_features import extract_html_features

DATA_PATH = "data/url_dataset.csv"
MODEL_PATH = "models/html_model.pkl"

FEATURE_ORDER = [
    "form_count",
    "password_input_count",
    "iframe_count",
    "external_link_count",
    "has_suspicious_words",
    "html_length"
]

def fetch_html(url):
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return response.text
    except requests.exceptions.RequestException:
        return ""

print("📥 Loading dataset...")
data = pd.read_csv(DATA_PATH)

if "url" not in data.columns or "label" not in data.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns")

print(f"✅ Loaded {len(data)} URLs")

print("🧠 Fetching HTML and extracting features...")

feature_rows = []

for url in data["url"]:
    html = fetch_html(url)
    features = extract_html_features(html)
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)
labels = data['label']

# 🔥 FIX: remove rows with NaN (failed HTML fetches)
combined = features_df.copy()
combined['label'] = labels

combined.dropna(inplace=True)

features_df = combined.drop(columns=['label'])
labels = combined['label']


print("✅ HTML feature extraction complete")

X_train, X_test, y_train, y_test = train_test_split(
    features_df,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("🚀 Training HTML Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("✅ Model training completed")

print("\n📊 Model Evaluation:")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\n💾 Model saved to {MODEL_PATH}")
print("🎉 HTML-based phishing model ready!")
