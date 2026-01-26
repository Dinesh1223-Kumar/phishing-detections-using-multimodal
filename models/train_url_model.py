import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from features.url_features import extract_url_features

# Step 1: Load the dataset
data = pd.read_csv("data/url_dataset.csv")

# Step 2: Convert URLs into numbers (features)
X = []
for url in data["url"]:
    features = extract_url_features(url)
    X.append(list(features.values()))

# Step 3: Labels (answers)
y = data["label"]

# Step 4: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Step 5: Save the trained model
joblib.dump(model, "models/url_model.pkl")

print("URL model trained and saved successfully")
