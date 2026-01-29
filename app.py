from flask import Flask, render_template, request
import joblib
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# Load Models ONCE
# ==============================
url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")

URL_FEATURES = list(url_model.feature_names_in_)
HTML_FEATURES = list(html_model.feature_names_in_)

# ==============================
# URL Feature Extraction
# ==============================
def extract_url_features(url):
    parsed = urlparse(url)
    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "has_https": int(parsed.scheme == "https"),
        "num_slashes": url.count("/"),
        "num_hyphens": url.count("-"),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "query_length": len(parsed.query),
        "path_length": len(parsed.path),
    }

    return [features.get(f, 0) for f in URL_FEATURES]

# ==============================
# HTML Feature Extraction
# ==============================
def extract_html_features(soup):
    features = {
        "num_forms": len(soup.find_all("form")),
        "num_links": len(soup.find_all("a")),
        "num_inputs": len(soup.find_all("input")),
        "num_iframes": len(soup.find_all("iframe")),
        "has_password": int(bool(soup.find("input", {"type": "password"}))),
        "title_length": len(soup.title.text) if soup.title else 0,
    }

    return [features.get(f, 0) for f in HTML_FEATURES]

# ==============================
# NLP Text Extraction
# ==============================
def extract_visible_text(soup):
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Fusion Prediction
# ==============================
def predict_phishing(url):
    try:
        response = requests.get(url, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return "Phishing", 95.0

    # --- URL ---
    url_vector = extract_url_features(url)
    url_prob = url_model.predict_proba([url_vector])[0][1]

    # --- HTML ---
    html_vector = extract_html_features(soup)
    html_prob = html_model.predict_proba([html_vector])[0][1]

    # --- NLP ---
    page_text = extract_visible_text(soup)
    text_vec = text_vectorizer.transform([page_text])
    text_prob = text_model.predict_proba(text_vec)[0][1]

    # --- Fusion ---
    final_score = (
        0.3 * url_prob +
        0.3 * html_prob +
        0.4 * text_prob
    )

    label = "Phishing" if final_score >= 0.5 else "Legitimate"
    return label, round(final_score * 100, 2)

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        url = request.form.get("url_input")
        result, confidence = predict_phishing(url)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
