import pandas as pd
import joblib
from features.url_features import extract_url_features

# Load trained model
model = joblib.load("models/url_model.pkl")

# Test URLs
urls = [
    "https://www.google.com",                      # real
    "https://www.amazon.in",                       # real
    "http://login-secure-paypal.com/verify",        # suspicious
    "http://secure-login@paypal.com.verify.info",  # very suspicious
    "http://192.168.1.1/login",                     # IP based
    "http://free-gift-card-win-now.com",            # phishing style
]


for url in urls:
    features = extract_url_features(url)
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




    print("FEATURES:", features)
    print("TYPE:", type(features))
    feature_vector = pd.DataFrame([[features[f] for f in FEATURE_ORDER]],
                              columns=FEATURE_ORDER)

    prediction = model.predict(feature_vector)


    print(url, "->", "Phishing" if prediction[0] == 1 else "Legitimate")
