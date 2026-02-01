from flask import Flask, render_template, request
import joblib
import os
import csv
from datetime import datetime
import requests

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
# Load models (once)
# ==============================
url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
network_model = joblib.load("models/network_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
behavioral_model = joblib.load("models/behavioral_model.pkl")

# ==============================
# Feature order
# ==============================
URL_FEATURE_ORDER = [
    "url_length", "count_dots", "count_hyphen", "has_at_symbol",
    "has_https", "has_login_word", "subdomain_count", "is_ip_address"
]

HTML_FEATURE_ORDER = [
    "form_count", "password_input_count", "iframe_count",
    "external_link_count", "has_suspicious_words", "html_length"
]

NETWORK_FEATURE_ORDER = [
    "domain_length", "num_subdomains", "has_ip_address",
    "dns_resolves", "uses_https"
]

# ====== 7 features for trained behavioral model ======
BEHAVIOR_FEATURE_ORDER = [
    "has_login_form", "password_input_count", "hidden_input_count",
    "has_submit_button", "has_urgent_words", "has_meta_refresh",
    "form_action_external"
]

# ==============================
# Logging all scans
# ==============================
def log_scan(url, label, probability, risk):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/scan_history.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "url", "label", "probability", "risk"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            url, label, probability, risk
        ])

# ==============================
# Dashboard stats
# ==============================
def get_dashboard_stats():
    stats = {"total": 0, "phishing": 0, "legit": 0, "suspicious": 0}
    log_file = "logs/scan_history.csv"
    if not os.path.isfile(log_file):
        return stats

    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            try:
                prob = float(row.get("probability", 0))
            except ValueError:
                prob = 0
            label = row.get("label", "Legitimate")

            if label == "Phishing":
                stats["phishing"] += 1
            elif 40 <= prob < 50:
                stats["suspicious"] += 1
            else:
                stats["legit"] += 1
    return stats

# ==============================
# Main phishing prediction
# ==============================
def predict_phishing(url):
    model_scores = {}
    reasons = []

    # -------- URL model --------
    url_features = extract_url_features(url)
    url_vector = [[url_features.get(f, 0) for f in URL_FEATURE_ORDER]]
    url_proba = url_model.predict_proba(url_vector)[0][1]
    model_scores["URL"] = round(url_proba, 2)
    if url_features.get("is_ip_address", 0):
        reasons.append("IP address used instead of domain name")
    if url_features.get("has_login_word", 0):
        reasons.append("Suspicious login-related word in URL")

    # -------- Network model --------
    net_features = extract_network_features(url)
    net_vector = [[net_features.get(f, 0) for f in NETWORK_FEATURE_ORDER]]
    net_proba = network_model.predict_proba(net_vector)[0][1]
    model_scores["Network"] = round(net_proba, 2)
    if net_features.get("dns_resolves", 1) == 0:
        reasons.append("Domain does not resolve via DNS")

    # -------- Optional HTML / NLP / Behavioral models --------
    html_content = None
    try:
        r = requests.get(url, timeout=5)
        if "text/html" in r.headers.get("Content-Type", ""):
            html_content = r.text
    except:
        html_content = None

    if html_content:
        # HTML features
        html_features = extract_html_features(html_content)
        html_vector = [[html_features.get(f, 0) for f in HTML_FEATURE_ORDER]]
        html_proba = html_model.predict_proba(html_vector)[0][1]
        model_scores["HTML"] = round(html_proba, 2)
        if html_features.get("password_input_count", 0) > 0:
            reasons.append("Password input field detected")

        # NLP
        text = extract_visible_text(html_content)
        cleaned = clean_text(text)
        text_vec = text_vectorizer.transform([cleaned])
        nlp_proba = text_model.predict_proba(text_vec)[0][1]
        model_scores["NLP"] = round(nlp_proba, 2)
        if any(w in cleaned for w in ["verify", "urgent", "confirm", "suspended"]):
            reasons.append("Urgent phishing language detected")

        # Behavioral
        beh_features = extract_behavioral_features(html_content, url)
        beh_vector = [[beh_features.get(f, 0) for f in BEHAVIOR_FEATURE_ORDER]]
        beh_proba = behavioral_model.predict_proba(beh_vector)[0][1]
        model_scores["Behavioral"] = round(beh_proba, 2)
        if beh_features.get("form_action_external", 0):
            reasons.append("Form submits data to an external domain")

    # -------- Fusion --------
    weights = {
        "URL": 0.25,
        "Network": 0.25,
        "HTML": 0.2 if html_content else 0,
        "NLP": 0.15 if html_content else 0,
        "Behavioral": 0.15 if html_content else 0
    }

    weighted_sum = sum(model_scores[m] * weights[m] for m in model_scores)
    total_weight = sum(weights[m] for m in model_scores)
    final_score = weighted_sum / total_weight if total_weight > 0 else 0

    probability = round(final_score * 100, 2)
    final_label = "Phishing" if probability >= 50 else "Legitimate"

    if probability >= 80:
        risk_level = "HIGH"
    elif probability >= 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Log scan
    log_scan(url, final_label, probability, risk_level)

    return {
        "final_label": final_label,
        "probability": probability,
        "risk_level": risk_level,
        "model_scores": model_scores,
        "reasons": list(set(reasons))
    }

# ==============================
# Routes
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            error = "Please enter a valid URL"
        else:
            result = predict_phishing(url)

    stats = get_dashboard_stats()
    return render_template("index.html", result=result, stats=stats, error=error)

# ==============================
# Run app
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
