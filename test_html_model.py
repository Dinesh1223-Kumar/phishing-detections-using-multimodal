import joblib
import requests
import pandas as pd

from features.html_features import extract_html_features

# =========================
# Load trained HTML model
# =========================
MODEL_PATH = "models/html_model.pkl"
model = joblib.load(MODEL_PATH)

print("✅ HTML model loaded successfully")


# =========================
# Test URL
# =========================
test_url = "https://github.com"   # you can change this later

print(f"\n🌐 Testing URL: {test_url}")


# =========================
# Fetch HTML safely
# =========================
try:
    response = requests.get(
        test_url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    html_content = response.text
except Exception as e:
    print("❌ Failed to fetch HTML:", e)
    exit()


# =========================
# Extract features
# =========================
features = extract_html_features(html_content)

features_df = pd.DataFrame([features])

print("\n🧠 Extracted HTML Features:")
print(features_df)


# =========================
# Prediction
# =========================
prediction = model.predict(features_df)[0]

if prediction == 1:
    print("\n🚨 RESULT: PHISHING WEBSITE")
else:
    print("\n✅ RESULT: LEGITIMATE WEBSITE")
