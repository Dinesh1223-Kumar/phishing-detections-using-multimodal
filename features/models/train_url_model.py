import joblib
import pandas as pd
import numpy as np
from features.url_features import extract_url_features

# ==============================
# Load Trained URL Model
# ==============================

try:
    model = joblib.load("models/url_model.pkl")
    print("✅ URL model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ==============================
# Test URL (Change if needed)
# ==============================

url = "https://github.com"   # You can modify this
print(f"\n🌐 Testing URL: {url}")

# ==============================
# Extract URL Features
# ==============================

try:
    features = extract_url_features(url)
    df = pd.DataFrame([features])

    print("\n🧠 Extracted URL Features:")
    print(df)

except Exception as e:
    print("❌ Feature extraction failed:", e)
    exit()

# ==============================
# Predict Using Model
# ==============================

try:
    # Convert to numpy array to avoid sklearn feature-name warning
    prediction = model.predict(np.array(df.values))[0]

    if prediction == 1:
        print("\n🚨 RESULT: PHISHING WEBSITE")
    else:
        print("\n✅ RESULT: LEGITIMATE WEBSITE")

except Exception as e:
    print("❌ Prediction failed:", e)