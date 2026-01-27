import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from features.url_features import extract_url_features


# =========================
# Configuration
# =========================
DATA_PATH = "data/url_dataset.csv"
MODEL_PATH = "models/url_model.pkl"

FEATURE_ORDER = [
    'url_length',
    'count_dots',
    'count_hyphen',
    'has_at_symbol',
    'has_https',
    'has_login_word',
    'subdomain_count',
    'is_ip_address'
]


# =========================
# 1. Load Dataset
# =========================
print("📥 Loading dataset...")
data = pd.read_csv(DATA_PATH)

if 'url' not in data.columns or 'label' not in data.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns")

print(f"✅ Loaded {len(data)} URLs")


# =========================
# 2. Feature Extraction
# =========================
print("🧠 Extracting URL features...")

feature_rows = []
for url in data['url']:
    feature_rows.append(extract_url_features(url))

features_df = pd.DataFrame(feature_rows)[FEATURE_ORDER]
labels = data['label']

print("✅ Feature extraction complete")


# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    features_df,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# =========================
# 4. Model Training
# =========================
print("🚀 Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("✅ Model training completed")


# =========================
# 5. Evaluation
# =========================
print("\n📊 Model Evaluation:")

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


# =========================
# 6. Save Model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\n💾 Model saved to {MODEL_PATH}")
print("🎉 URL-based phishing model ready!")
