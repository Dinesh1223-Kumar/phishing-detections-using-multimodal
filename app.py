from flask import Flask, render_template, request
import joblib, os, csv, socket, ssl
from datetime import datetime
import requests
import whois
import tldextract

from features.url_features import extract_url_features
from features.html_features import extract_html_features
from features.network_features import extract_network_features
from features.behavioral_features import extract_behavioral_features
from features.text_extractor import extract_visible_text
from features.text_preprocessing import clean_text

app = Flask(__name__)

# ================== Load models ==================
url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
network_model = joblib.load("models/network_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
behavioral_model = joblib.load("models/behavioral_model.pkl")

# ================== Feature Order ==================
URL_FEATURE_ORDER = [
    "url_length","count_dots","count_hyphen","has_at_symbol",
    "has_https","has_login_word","subdomain_count","is_ip_address"
]

HTML_FEATURE_ORDER = [
    "form_count","password_input_count","iframe_count",
    "external_link_count","has_suspicious_words","html_length"
]

NETWORK_FEATURE_ORDER = [
    "domain_length","num_subdomains","has_ip_address",
    "dns_resolves","uses_https"
]

BEHAVIOR_FEATURE_ORDER = [
    "has_login_form","password_input_count","hidden_input_count",
    "has_submit_button","has_urgent_words","has_meta_refresh",
    "form_action_external"
]

# ================== Ensure logs ==================
def ensure_file(path, headers):
    os.makedirs("logs", exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

ensure_file("logs/scan_history.csv",
            ["timestamp","url","label","probability","risk"])

ensure_file("logs/phishing_urls.csv",
            ["timestamp","url","probability","risk"])

ensure_file("logs/legit_urls.csv",
            ["timestamp","url","probability"])

ensure_file("logs/uncertain_predictions.csv",
            ["timestamp","url","probability"])

# ================== Logging ==================
def log_scan(url, label, probability, risk):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("logs/scan_history.csv","a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([now,url,label,probability,risk])

    if label == "Phishing":
        with open("logs/phishing_urls.csv","a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([now,url,probability,risk])

    elif label == "Legitimate":
        with open("logs/legit_urls.csv","a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([now,url,probability])

    else:
        with open("logs/uncertain_predictions.csv","a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([now,url,probability])

# ================== Dashboard helpers ==================
def count_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

def get_dashboard_stats():
    return {
        "total": count_rows("logs/scan_history.csv"),
        "phishing": count_rows("logs/phishing_urls.csv"),
        "legit": count_rows("logs/legit_urls.csv"),
        "suspicious": count_rows("logs/uncertain_predictions.csv")
    }

def get_recent_scans(limit=4):
    path = "logs/scan_history.csv"
    if not os.path.isfile(path):
        return []

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[-limit:]

    return [{
        "url": r["url"],
        "label": r["label"],
        "date": r["timestamp"]
    } for r in reversed(rows)]

# ================== Domain intelligence ==================
from urllib.parse import urlparse

def get_domain_intelligence(url):

    data = {
        "domain_age": "Unknown",
        "ssl_status": "Unknown",
        "country": "Unknown",
        "registrar": "Unknown"
    }

    try:
        # Ensure scheme
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        parsed = urlparse(url)
        domain = parsed.netloc

        # Remove www
        if domain.startswith("www."):
            domain = domain[4:]

        # ================= SSL CHECK =================
        if url.startswith("https"):
            data["ssl_status"] = "Valid (HTTPS)"
        else:
            data["ssl_status"] = "No HTTPS"

        # ================= WHOIS CHECK =================
        try:
            w = whois.whois(domain)

            # Domain age
            if w.creation_date:
                created = w.creation_date
                if isinstance(created, list):
                    created = created[0]

                age_days = (datetime.now() - created).days
                data["domain_age"] = age_days

            # Registrar
            if w.registrar:
                data["registrar"] = w.registrar

            # Country
            if w.country:
                data["country"] = w.country

        except Exception as e:
            print("WHOIS error:", e)

    except Exception as e:
        print("Domain Intelligence Error:", e)

    return data


# ================== Prediction ==================
def predict_phishing(url):

     # ✅ Ensure URL has scheme
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    raw_scores = {}

    uf = extract_url_features(url)
    raw_scores["URL"] = url_model.predict_proba(
        [[uf[f] for f in URL_FEATURE_ORDER]]
    )[0][1]

    nf = extract_network_features(url)
    raw_scores["Network"] = network_model.predict_proba(
        [[nf[f] for f in NETWORK_FEATURE_ORDER]]
    )[0][1]

    html = None
    try:
        r = requests.get(url, timeout=5)
        if "text/html" in r.headers.get("Content-Type",""):
            html = r.text
    except:
        pass

    if html:
        hf = extract_html_features(html)
        raw_scores["HTML"] = html_model.predict_proba(
            [[hf[f] for f in HTML_FEATURE_ORDER]]
        )[0][1]

        text = clean_text(extract_visible_text(html))
        raw_scores["NLP"] = text_model.predict_proba(
            text_vectorizer.transform([text])
        )[0][1]

        bf = extract_behavioral_features(html, url)
        raw_scores["Behavioral"] = behavioral_model.predict_proba(
            [[bf[f] for f in BEHAVIOR_FEATURE_ORDER]]
        )[0][1]

    weights = {"URL":0.25,"Network":0.25,"HTML":0.2,"NLP":0.15,"Behavioral":0.15}
    final = sum(raw_scores[k]*weights[k] for k in raw_scores) / sum(weights[k] for k in raw_scores)
    probability = round(final * 100, 2)

    if probability >= 80:
        label, risk = "Phishing", "HIGH"
    elif probability >= 50:
        label, risk = "Suspicious", "MEDIUM"
    else:
        label, risk = "Legitimate", "LOW"

    analysis_scores = {
        "URL Analysis": int(raw_scores.get("URL", 0) * 100),
        "Network Reputation": int(raw_scores.get("Network", 0) * 100),
        "HTML Structure": int(raw_scores.get("HTML", 0) * 100),
        "Language Analysis": int(raw_scores.get("NLP", 0) * 100),
        "Behavioral Signals": int(raw_scores.get("Behavioral", 0) * 100)
    }

    domain_info = get_domain_intelligence(url)
    log_scan(url, label, probability, risk)

    return {
        "final_label": label,
        "risk_level": risk,
        "probability": probability,
        "analysis_scores": analysis_scores,
        **domain_info
    }

# ================== ROUTES ==================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            result = predict_phishing(url)

    return render_template(
        "index.html",
        result=result,
        stats=get_dashboard_stats(),
        recent_scans=get_recent_scans()
    )

@app.route("/total-scans")
def total_scans():
    with open("logs/scan_history.csv", newline="", encoding="utf-8") as f:
        scans = list(csv.DictReader(f))
    return render_template("total_scans.html", scans=scans)

@app.route("/phishing")
def phishing():
    with open("logs/phishing_urls.csv", newline="", encoding="utf-8") as f:
        scans = list(csv.DictReader(f))
    return render_template("phishing.html", scans=scans)

@app.route("/legitimate")
def legitimate():
    with open("logs/legit_urls.csv", newline="", encoding="utf-8") as f:
        scans = list(csv.DictReader(f))
    return render_template("legitimate.html", scans=scans)

@app.route("/suspicious")
def suspicious():
    with open("logs/uncertain_predictions.csv", newline="", encoding="utf-8") as f:
        scans = list(csv.DictReader(f))
    return render_template("suspicious.html", scans=scans)
# ================== Run ==================
if __name__ == "__main__":
    app.run(debug=True)
