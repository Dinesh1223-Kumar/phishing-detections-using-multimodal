# ==============================
# Required Imports
# ==============================
import joblib
import nltk

from features.text_extractor import extract_visible_text
from features.text_preprocessing import clean_text


# ==============================
# One-time NLTK setup (safe check)
# ==============================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# ==============================
# Load NLP model & vectorizer
# (Loaded once for performance)
# ==============================
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")


def predict_nlp_from_html(html_content):
    """
    Input: raw HTML
    Output: 1 (Phishing) or 0 (Legit)
    """

    # 1️⃣ Extract visible text
    visible_text = extract_visible_text(html_content)

    # 2️⃣ Clean extracted text
    cleaned_text = clean_text(visible_text)

    # 3️⃣ Vectorize text
    text_vec = text_vectorizer.transform([cleaned_text])

    # 4️⃣ Predict
    prediction = text_model.predict(text_vec)[0]

    return int(prediction)


# ==============================
# Optional: Direct test run
# ==============================
if __name__ == "__main__":
    sample_html = """
    <html>
        <body>
            <h2>Account Verification Required</h2>
            <p>Your account has been suspended. Click below to verify.</p>
            <a href="http://fake-login.com">Verify Now</a>
        </body>
    </html>
    """

    result = predict_nlp_from_html(sample_html)
    print("Prediction:", "Phishing 🚨" if result == 1 else "Legit ✅")
