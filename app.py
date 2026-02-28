from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
import joblib, os, csv
from datetime import datetime
import requests
import whois
from urllib.parse import urlparse

from features.url_features import extract_url_features
from features.html_features import extract_html_features
from features.network_features import extract_network_features
from features.behavioral_features import extract_behavioral_features
from features.text_extractor import extract_visible_text
from features.text_preprocessing import clean_text

# ================== Load Models ==================
cnn_model = tf.keras.models.load_model("phishing_cnn_model.keras")

url_model = joblib.load("models/url_model.pkl")
html_model = joblib.load("models/html_model.pkl")
network_model = joblib.load("models/network_model.pkl")
text_model = joblib.load("models/text_model.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
behavioral_model = joblib.load("models/behavioral_model.pkl")

app = Flask(__name__)

# ================== Feature Orders ==================
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

# ================== Screenshot ==================
def capture_screenshot(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1280,1024")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    try:
        driver.get(url)
        driver.save_screenshot("temp_screenshot.png")
        driver.quit()
        return "temp_screenshot.png"
    except:
        driver.quit()
        return None

# ================== CNN ==================
def predict_visual_phishing(image_path):
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(img_array, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print("CNN Error:", e)
        return 0.5
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)[0][0]
    return float(prediction)

# ================== Logging ==================
os.makedirs("logs", exist_ok=True)

def ensure_file(path, headers):
    if not os.path.isfile(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

ensure_file("logs/scan_history.csv",
            ["timestamp","url","label","probability","risk"])

def log_scan(url, label, probability, risk):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/scan_history.csv","a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([now,url,label,probability,risk])

def load_stats():
    stats = {"total":0,"phishing":0,"legit":0,"suspicious":0}
    recent = []

    try:
        with open("logs/scan_history.csv","r",encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            stats["total"] = len(reader)

            for row in reader:
                if row["label"] == "Phishing":
                    stats["phishing"] += 1
                elif row["label"] == "Legitimate":
                    stats["legit"] += 1
                elif row["label"] == "Suspicious":
                    stats["suspicious"] += 1

            for row in reader[-5:][::-1]:
                recent.append({
                    "url": row["url"],
                    "label": row["label"],
                    "date": row["timestamp"]
                })
    except:
        pass

    return stats, recent

# ================== Domain Intelligence ==================
def get_domain_intelligence(url):
    data = {
        "domain_age": "Unknown",
        "ssl_status": "Unknown",
        "country": "Unknown",
        "registrar": "Unknown"
    }

    try:
        if not url.startswith(("http://","https://")):
            url = "http://" + url

        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.","")

        data["ssl_status"] = "Valid (HTTPS)" if url.startswith("https") else "No HTTPS"

        try:
            w = whois.whois(domain)
            if w.creation_date:
                created = w.creation_date
                if isinstance(created, list):
                    created = created[0]
                data["domain_age"] = (datetime.now() - created).days

            if w.registrar:
                data["registrar"] = w.registrar
            if w.country:
                data["country"] = w.country
        except:
            pass
    except:
        pass

    return data

# ================== Prediction ==================
def predict_phishing(url):

    if not url.startswith(("http://","https://")):
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

    screenshot_path = capture_screenshot(url)
    raw_scores["Visual"] = predict_visual_phishing(screenshot_path) if screenshot_path else 0.5

    weights = {
        "URL":0.2,"Network":0.2,"HTML":0.15,
        "NLP":0.15,"Behavioral":0.15,"Visual":0.15
    }

    final = sum(raw_scores[k]*weights[k] for k in raw_scores) / \
            sum(weights[k] for k in raw_scores)

    probability = round(final*100,2)

    if probability >= 80:
        label,risk = "Phishing","HIGH"
    elif probability >= 50:
        label,risk = "Suspicious","MEDIUM"
    else:
        label,risk = "Legitimate","LOW"

    log_scan(url,label,probability,risk)

    analysis_scores = {
        "URL Analysis": int(raw_scores.get("URL",0)*100),
        "Network Reputation": int(raw_scores.get("Network",0)*100),
        "HTML Structure": int(raw_scores.get("HTML",0)*100),
        "Language Analysis": int(raw_scores.get("NLP",0)*100),
        "Behavioral Signals": int(raw_scores.get("Behavioral",0)*100),
        "Visual Analysis": int(raw_scores.get("Visual",0)*100)
    }

    return {
        "final_label":label,
        "risk_level":risk,
        "probability":probability,
        "analysis_scores":analysis_scores,
        **get_domain_intelligence(url)
    }

# ================== Routes ==================
@app.route("/",methods=["GET","POST"])
def index():
    result=None
    if request.method=="POST":
        url=request.form.get("url","").strip()
        if url:
            result=predict_phishing(url)

    stats,recent_scans=load_stats()
    return render_template("index.html",
                           result=result,
                           stats=stats,
                           recent_scans=recent_scans)

@app.route("/total_scans")
def total_scans():
    stats,_=load_stats()
    return f"Total Scans: {stats['total']}"

@app.route("/phishing")
def phishing():
    stats,_=load_stats()
    return f"Phishing URLs: {stats['phishing']}"

@app.route("/legitimate")
def legitimate():
    stats,_=load_stats()
    return f"Legitimate URLs: {stats['legit']}"

@app.route("/suspicious")
def suspicious():
    stats,_=load_stats()
    return f"Suspicious URLs: {stats['suspicious']}"

# ================== Run ==================
if __name__=="__main__":
    app.run(debug=True)