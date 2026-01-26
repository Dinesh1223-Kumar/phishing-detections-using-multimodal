from flask import Flask
from features.url_features import extract_url_features

app = Flask(__name__)

@app.route("/")
def home():
    test_url = "http://example-login-secure.com"
    features = extract_url_features(test_url)
    return str(features)

if __name__ == "__main__":
    app.run(debug=True)
