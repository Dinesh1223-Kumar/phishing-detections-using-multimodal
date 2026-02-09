# EXPERIMENTAL: Train a model using LOGISTIC REGRESSION HTML features (not deployed)
# This is for algorithm comparison experiments only
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from experimental_features.html_features_exp import extract_html_features_exp

DATA_PATH = "data/url_dataset.csv"

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

features = []
labels = []

print("🧠 Extracting EXPERIMENTAL HTML features...")

for url, label in zip(data["url"], data["label"]):
    html = fetch_html(url)
    feats = extract_html_features_exp(html)
    if feats is not None:
        features.append(feats)
        labels.append(label)

X = pd.DataFrame(features)
y = pd.Series(labels)

print("✅ Feature extraction completed")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("🚀 Training Logistic Regression (EXPERIMENTAL)...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("✅ Training completed")

y_pred = model.predict(X_test)

print("\n📊 Experimental Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n🧪 Experimental model training finished (not deployed)")
