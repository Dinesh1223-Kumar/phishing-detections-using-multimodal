import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from features.url_features import extract_url_features

# 1️⃣ Load dataset
data = pd.read_csv("data/url_dataset.csv")

# 2️⃣ Extract features
feature_rows = []
for url in data['url']:
    feature_rows.append(extract_url_features(url))

features_df = pd.DataFrame(feature_rows)

# 3️⃣ Add labels
features_df['label'] = data['label']

print("\nExtracted Features:")
print(features_df)

# 4️⃣ Split dataset
X = features_df.drop('label', axis=1)
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Evaluate
print("\nTraining Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# 7️⃣ Save model
joblib.dump(model, "models/url_model.pkl")
print("\n✅ Model saved to models/url_model.pkl")
