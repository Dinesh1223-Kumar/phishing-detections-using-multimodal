import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1️⃣ Load cleaned dataset
data = pd.read_csv("data/behavioral_dataset_clean.csv")

# 2️⃣ Split features and label
X = data.drop("label", axis=1)
y = data["label"]

# 3️⃣ Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

# 4️⃣ Save model
joblib.dump(model, "models/behavioral_model.pkl")

print("✅ Behavioral model trained and saved successfully")
