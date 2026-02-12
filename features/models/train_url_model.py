import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

from features.url_features import extract_url_features

# Load dataset
data = pd.read_csv("data/url_dataset.csv")

print("Original size:", len(data))

# Remove rows where label is NaN
data = data.dropna(subset=["label"])

# Remove empty string labels
data = data[data["label"] != ""]

# Convert label to numeric (force conversion)
data["label"] = pd.to_numeric(data["label"], errors="coerce")

# Drop rows that failed conversion
data = data.dropna(subset=["label"])

# Convert to int
data["label"] = data["label"].astype(int)

print("Cleaned size:", len(data))
print("Unique labels:", data["label"].unique())

# Extract features
X = []
for url in data["url"]:
    X.append(list(extract_url_features(url).values()))

y = data["label"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save
joblib.dump(model, "models/url_model.pkl")

print("URL model trained and saved successfully")
