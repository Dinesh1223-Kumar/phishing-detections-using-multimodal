import pandas as pd
import requests

from features.behavioral_features import extract_behavioral_features


# 1️⃣ Load URL dataset
data = pd.read_csv("data/url_dataset.csv")

feature_rows = []
labels = []

for _, row in data.iterrows():
    url = row["url"]
    label = row["label"]

    try:
        # 2️⃣ Fetch HTML
        response = requests.get(url, timeout=8)
        html = response.text

        # 3️⃣ Extract behavioral features
        features = extract_behavioral_features(html, url)

    except Exception:
        # Safe fallback if URL fails
        features = {
            "has_login_form": 0,
            "password_input_count": 0,
            "has_submit_button": 0,
            "hidden_input_count": 0,
            "has_urgent_words": 0,
            "has_meta_refresh": 0,
            "form_action_external": 0
        }

    feature_rows.append(features)
    labels.append(label)


# 4️⃣ Create DataFrame
X = pd.DataFrame(feature_rows)
X["label"] = labels

# 5️⃣ Save dataset
X.to_csv("data/behavioral_dataset.csv", index=False)

print("✅ Behavioral dataset created successfully")
