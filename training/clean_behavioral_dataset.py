import pandas as pd

# 1️⃣ Load behavioral dataset
data = pd.read_csv("data/behavioral_dataset.csv")

print("Before cleaning:")
print(data.isnull().sum())

# 2️⃣ Drop rows where label is missing
data = data.dropna(subset=["label"])

# 3️⃣ Fill missing feature values with 0
feature_cols = [col for col in data.columns if col != "label"]
data[feature_cols] = data[feature_cols].fillna(0)

print("\nAfter cleaning:")
print(data.isnull().sum())

# 4️⃣ Save cleaned dataset
data.to_csv("data/behavioral_dataset_clean.csv", index=False)

print("✅ Behavioral dataset cleaned and saved")
