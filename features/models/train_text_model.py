import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

from features.text_preprocessing import clean_text


# =========================
# 1️⃣ Load NLP dataset
# =========================
# Dataset format:
# text,label
# "Verify your account...",1
# "Welcome to our homepage",0
data = pd.read_csv("data/text_dataset.csv")


# =========================
# 2️⃣ Clean and preprocess text
# =========================
# Apply custom text cleaning function:
# - lowercase
# - remove URLs
# - remove symbols
# - normalize spaces
data["clean_text"] = data["text"].apply(clean_text)


# =========================
# 3️⃣ Split features and labels
# =========================
X = data["clean_text"]   # input text
y = data["label"]        # target labels (0 = legit, 1 = phishing)


# =========================
# 4️⃣ Convert text to numerical features (TF-IDF)
# =========================
# TF-IDF measures how important a word is in a document
vectorizer = TfidfVectorizer(
    max_features=5000,      # limit vocabulary size
    stop_words="english"    # remove common words like 'the', 'is'
)

# Learn vocabulary + transform text
X_vec = vectorizer.fit_transform(X)


# =========================
# 5️⃣ Train Machine Learning model
# =========================
# RandomForest is robust and works well with TF-IDF vectors
model = RandomForestClassifier(
    random_state=42
)

# Train the classifier
model.fit(X_vec, y)


# =========================
# 6️⃣ Save trained model and vectorizer
# =========================
# These will be loaded during prediction in Flask app
joblib.dump(model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/text_vectorizer.pkl")


# =========================
# 7️⃣ Confirmation message
# =========================
print("✅ NLP model trained and saved successfully")
