from flask import Flask, render_template, request
import joblib
from features.url_features import extract_url_features

app = Flask(__name__)

# Load the trained model using joblib
url_model = joblib.load("models/url_model.pkl")  # <-- FIXED

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url_input = request.form["url_input"]

        # Extract features from the URL
        features = extract_url_features(url_input)
        feature_values = [list(features.values())]  # Convert to 2D list for sklearn

        # Predict using the model
        prediction = url_model.predict(feature_values)[0]

        # Map prediction to readable text
        result = "Phishing" if prediction == 1 else "Legit"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
