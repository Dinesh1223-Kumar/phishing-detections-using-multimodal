import requests
from features.html_features import extract_html_features

def fetch_html(url):
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return response.text
    except requests.exceptions.RequestException:
        return ""

url = "https://example.com"

html = fetch_html(url)

features = extract_html_features(html)
print(features)
