# test_nlp_model.py

import sys
import os
import joblib

# -----------------------------
# Add 'features' folder to Python path
# -----------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))

# Now we can import from features
from text_extractor import extract_visible_text
from text_preprocessing import clean_text

# -----------------------------
# 1️⃣ Load the trained model & vectorizer
# -----------------------------
try:
    model = joblib.load("models/text_model.pkl")
    vectorizer = joblib.load("models/text_vectorizer.pkl")
    print("✅ NLP model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("❌ Model or vectorizer not found. Train the NLP model first.")
    exit()

# -----------------------------
# 2️⃣ Function to predict text from HTML
# -----------------------------
def predict_nlp(html_content):
    """
    Input: raw HTML string
    Output: 'Phishing' or 'Legit'
    """
    # Extract visible text from HTML
    visible_text = extract_visible_text(html_content)

    # Clean text (lowercase, remove URLs, symbols, etc.)
    cleaned_text = clean_text(visible_text)

    # Convert to TF-IDF vector
    text_vec = vectorizer.transform([cleaned_text])

    # Predict
    pred = model.predict(text_vec)[0]

    return "Phishing" if pred == 1 else "Legit"

# -----------------------------
# 3️⃣ Test with sample HTMLs
# -----------------------------
if __name__ == "__main__":
    # Sample phishing HTML
    phishing_html = """
    <html><body>
    Your account has been locked! Click <a href='http://fakebank.com'>here</a> to verify immediately.
    </body></html>
    """

    # Sample legit HTML
    legit_html = """
    <html><body>
    Welcome to our official company website. Explore our services and contact support for assistance.
    </body></html>
    """

    print("\nTesting NLP model:\n")
    print("Sample phishing HTML prediction:", predict_nlp(phishing_html))
    print("Sample legit HTML prediction:", predict_nlp(legit_html))
