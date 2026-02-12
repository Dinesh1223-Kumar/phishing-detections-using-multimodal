import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from features.url_features import extract_url_features

# Load dataset
data = pd.read_csv("data/url_dataset.csv")

print("Original dataset size:", len(data))

# Remove completely empty rows
data = data.dropna()

# Remove rows where label is empty string
data = data[data["label"] != ""]

# If labels are text convert them
if data["label"].dtype == object:
    data["label"] = data["label"].str.lower()
    data["label"] = data["label"].map({
        "phishing": 1,
        "legitimate": 0,
        "0": 0,
        "1": 1
    })

# Remove any rows that became NaN after mapping
data = data.dropna(subset=["label"])

# Convert to int
data["label"] = data["label"].astype(int)

print("Cleaned dataset size:", len(data))
print("Unique labels:", data["label"].unique())

# Extract features
feature_list = []
for url in data["url"]:
    feature_list.append(extract_url_features(url))

X = pd.DataFrame(feature_list)

# Fill any feature NaN just in case
X = X.fillna(0)

y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, "models/url_model.pkl")

print("URL Model trained and saved successfully!")
