import pickle
import numpy as np
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

# -----------------------------
# Load URL and HTML models safely
# -----------------------------
try:
    with open("models/url_model.pkl", "rb") as f:
        url_model = pickle.load(f)
except Exception as e:
    print(f"Failed to load URL model: {e}")
    url_model = None

try:
    with open("models/html_model.pkl", "rb") as f:
        html_model = pickle.load(f)
except Exception as e:
    print(f"Failed to load HTML model: {e}")
    html_model = None

# -----------------------------
# Define expected features
# -----------------------------
URL_FEATURE_ORDER = ['length', 'num_dots', 'has_https']  # Example features
HTML_FEATURE_ORDER = ['title_length', 'num_forms', 'num_links']  # Example features

# -----------------------------
# URL feature extraction
# -----------------------------
def extract_url_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['has_https'] = int(url.startswith('https'))
    return features

# -----------------------------
# HTML feature extraction
# -----------------------------
def extract_html_features(url):
    features = {}
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        features['title_length'] = len(soup.title.text) if soup.title else 0
        features['num_forms'] = len(soup.find_all('form'))
        features['num_links'] = len(soup.find_all('a'))
    except Exception as e:
        # If page cannot be fetched, set defaults
        features['title_length'] = 0
        features['num_forms'] = 0
        features['num_links'] = 0
    return features

# -----------------------------
# Fusion prediction
# -----------------------------
def predict_phishing(url):
    if url_model is None or html_model is None:
        raise ValueError("Models not loaded.")

    # --- URL prediction ---
    url_features = extract_url_features(url)
    url_vector = [url_features.get(f, 0) for f in URL_FEATURE_ORDER]
    url_pred = url_model.predict([url_vector])[0]

    # --- HTML prediction ---
    html_features = extract_html_features(url)
    html_vector = [html_features.get(f, 0) for f in HTML_FEATURE_ORDER]
    html_pred = html_model.predict([html_vector])[0]

    # --- Fusion ---
    return 1 if url_pred == 1 or html_pred == 1 else 0
