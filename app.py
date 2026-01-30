from flask import Flask, render_template, request
import joblib

from features.url_features import extract_url_features
from features.html_features import extract_html_features
from features.network_features import extract_network_features
from features.text_extractor import extract_visible_text
from features.text_preprocessing import clean_text

app = Flask(__name__)

# ==============================
# Load models ONCE
# ==============================
url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
network_model = joblib.load("models/network_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")

# ==============================
# Feature orders (MATCH TRAINING)
# ==============================
URL_FEATURE_ORDER = [
    'url_length',
    'count_dots',
    'count_hyphen',
    'has_at_symbol',
    'has_https',
    'has_login_word',
    'subdomain_count',
    'is_ip_address'
]

HTML_FEATURE_ORDER = [
    'form_count',
    'password_input_count',
    'iframe_count',
    'external_link_count',
    'has_suspicious_words',
    'html_length'
]

NETWORK_FEATURE_ORDER = [
    'domain_length',
    'num_subdomains',
    'has_ip_address',
    'dns_resolves',
    'uses_https'
]

# ==============================
# Fusion prediction
# ==============================
def predict_phishing(url, html_content=""):
    votes = []

    # -------- URL MODEL --------
    url_features = extract_url_features(url)
    url_vector = [[url_features[f] for f in URL_FEATURE_ORDER]]
    votes.append(url_model.predict(url_vector)[0])

    # -------- HTML MODEL --------
    if html_content:
        html_features = extract_html_features(html_content)
        html_vector = [[html_features[f] for f in HTML_FEATURE_ORDER]]
        votes.append(html_model.predict(html_vector)[0])

    # -------- NETWORK MODEL --------
    net_features = extract_network_features(url)
    net_vector = [[net_features[f] for f in NETWORK_FEATURE_ORDER]]
    votes.append(network_model.predict(net_vector)[0])

    # -------- NLP MODEL --------
    if html_content:
        text = extract_visible_text(html_content)
        cleaned = clean_text(text)
        text_vec = text_vectorizer.transform([cleaned])
        votes.append(text_model.predict(text_vec)[0])

    # -------- FINAL DECISION (Majority vote)
    final_prediction = 1 if sum(votes) >= 2 else 0
    return final_prediction

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        html = request.form.get("html", "")

        if not url:
            error = "Please enter a valid URL"
        else:
            result = predict_phishing(url, html)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
