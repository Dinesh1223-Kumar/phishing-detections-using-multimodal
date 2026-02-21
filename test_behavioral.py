import joblib
import pandas as pd
import requests
import numpy as np
from features.behavioral_features import extract_behavioral_features

# ==============================
# Load Trained Behavioral Model
# ==============================

try:
    model = joblib.load("models/behavioral_model.pkl")
    print("✅ Behavioral model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ==============================
# Test URL (Change if needed)
# ==============================

url = "https://github.com"   # You can modify this for testing
print(f"\n🌐 Testing URL: {url}")

# ==============================
# Fetch Webpage
# ==============================

try:
    response = requests.get(url, timeout=10)
    html_content = response.text
except Exception as e:
    print("❌ Error fetching webpage:", e)
    exit()

# ==============================
# Extract Behavioral Features
# ==============================

try:
    features = extract_behavioral_features(html_content)
    df = pd.DataFrame([features])
    
    print("\n🧠 Extracted Behavioral Features:")
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