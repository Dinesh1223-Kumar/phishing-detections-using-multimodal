from flask import Flask, render_template, request
import joblib

# ==============================
# Feature extractors
# ==============================
from features.url_features import extract_url_features
from features.html_features import extract_html_features
from features.network_features import extract_network_features
from features.behavioral_features import extract_behavioral_features
from features.text_extractor import extract_visible_text
from features.text_preprocessing import clean_text

app = Flask(__name__)

# ==============================
# Load ALL models (once)
# ==============================
url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
network_model = joblib.load("models/network_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
behavioral_model = joblib.load("models/behavioral_model.pkl")

# ==============================
# Feature order (MUST match training)
# ==============================
URL_FEATURE_ORDER = [
    "url_length",
    "count_dots",
    "count_hyphen",
    "has_at_symbol",
    "has_https",
    "has_login_word",
    "subdomain_count",
    "is_ip_address"
]

HTML_FEATURE_ORDER = [
    "form_count",
    "password_input_count",
    "iframe_count",
    "external_link_count",
    "has_suspicious_words",
    "html_length"
]

NETWORK_FEATURE_ORDER = [
    "domain_length",
    "num_subdomains",
    "has_ip_address",
    "dns_resolves",
    "uses_https"
]

BEHAVIOR_FEATURE_ORDER = [
    "has_login_form",
    "password_input_count",
    "hidden_input_count",
    "has_submit_button",
    "has_urgent_words",
    "has_meta_refresh",
    "form_action_external",
    "multiple_forms",
    "many_inputs"
]

# ==============================
# Fusion logic (decision-level)
# ==============================
def predict_phishing(url, html_content=""):
    votes = []
    model_votes = {}

    # -------- URL MODEL --------
    url_features = extract_url_features(url)
    url_vector = [[url_features[f] for f in URL_FEATURE_ORDER]]
    url_pred = url_model.predict(url_vector)[0]
    votes.append(url_pred)
    model_votes["URL Model"] = url_pred

    # -------- HTML MODEL --------
    if html_content:
        html_features = extract_html_features(html_content)
        html_vector = [[html_features[f] for f in HTML_FEATURE_ORDER]]
        html_pred = html_model.predict(html_vector)[0]
        votes.append(html_pred)
        model_votes["HTML Model"] = html_pred

    # -------- NETWORK MODEL --------
    net_features = extract_network_features(url)
    net_vector = [[net_features[f] for f in NETWORK_FEATURE_ORDER]]
    net_pred = network_model.predict(net_vector)[0]
    votes.append(net_pred)
    model_votes["Network Model"] = net_pred

    # -------- NLP MODEL --------
    if html_content:
        text = extract_visible_text(html_content)
        cleaned = clean_text(text)
        text_vec = text_vectorizer.transform([cleaned])
        nlp_pred = text_model.predict(text_vec)[0]
        votes.append(nlp_pred)
        model_votes["NLP Model"] = nlp_pred

    # -------- BEHAVIORAL MODEL --------
    if html_content:
        beh_features = extract_behavioral_features(html_content, url)
        beh_vector = [[beh_features[f] for f in BEHAVIOR_FEATURE_ORDER]]
        beh_pred = behavioral_model.predict(beh_vector)[0]
        votes.append(beh_pred)
        model_votes["Behavioral Model"] = beh_pred

    # -------- FINAL DECISION --------
    score = sum(votes) / len(votes)
    final_prediction = 1 if score >= 0.5 else 0
    confidence = round(score * 100, 2)

    return final_prediction, confidence, model_votes

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    model_votes = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        html = request.form.get("html", "")

        if not url:
            error = "Please enter a valid URL"
        else:
            result, confidence, model_votes = predict_phishing(url, html)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        model_votes=model_votes,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
