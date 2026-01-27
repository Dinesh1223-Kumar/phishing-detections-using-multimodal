from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

from features.url_features import extract_url_features

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "url_model.pkl")

# Load trained model ONCE
model = joblib.load(MODEL_PATH)

# ---------------- FLASK APP ----------------
app = Flask(__name__)

FEATURE_ORDER = [
    'url_length',
    'count_dots',
    'count_hyphen',
    'has_at_symbol',
    'has_https',
    'has_login_word',
    'subdomain_count',
    'is_ip_address'
]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    url = ""

    if request.method == "POST":
        url = request.form["url_input"]

        features = extract_url_features(url)

        feature_vector = pd.DataFrame(
            [[features[f] for f in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        prediction = model.predict(feature_vector)[0]
        result = "Phishing" if prediction == 1 else "Legitimate"

    return render_template("index.html", result=result, url=url)

if __name__ == "__main__":
    app.run(debug=True)
