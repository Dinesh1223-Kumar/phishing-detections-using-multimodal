import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from features.network_features import extract_network_features

# Load dataset
data = pd.read_csv("data/url_dataset.csv")

feature_rows = []
labels = []

# Iterate through dataset
for _, row in data.iterrows():
    label = row["label"]

    # 🚨 Skip rows with missing labels
    if pd.isna(label):
        continue

    # Extract network features
    features = extract_network_features(row["url"])

    # Store features and labels safely
    feature_rows.append(features)
    labels.append(int(label))  # ensure numeric (0/1)

# Convert to DataFrame
X = pd.DataFrame(feature_rows)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/network_model.pkl")

print("✅ Network model trained and saved successfully")
