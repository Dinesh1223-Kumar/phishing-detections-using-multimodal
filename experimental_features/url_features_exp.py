import re
import tldextract
from urllib.parse import urlparse

def extract_url_features(url):
    features = {}

    # Basic Features
    features['url_length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphen'] = url.count('-')
    features['count_slash'] = url.count('/')
    features['count_question'] = url.count('?')
    features['count_equal'] = url.count('=')

    # Special Symbols
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_https'] = 1 if url.startswith("https") else 0

    # Phishing Keywords
    phishing_words = ['login', 'verify', 'secure', 'update', 'account', 'bank', 'confirm']
    features['has_phishing_word'] = 1 if any(word in url.lower() for word in phishing_words) else 0

    # Subdomain Count (More accurate using tldextract)
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    features['subdomain_count'] = len(subdomain.split('.')) if subdomain else 0

    # IP Address Detection
    features['is_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    # URL Path Length
    parsed = urlparse(url)
    features['path_length'] = len(parsed.path)

    # Digit Count
    features['digit_count'] = sum(c.isdigit() for c in url)

    return features
