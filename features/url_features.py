import re

def extract_url_features(url):
    features = {}

    # Existing features
    features['url_length'] = len(url)
    features['count_dots'] = url.count('.')
    features['count_hyphen'] = url.count('-')
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_https'] = 1 if url.startswith("https") else 0

    # NEW feature 1: phishing keywords
    phishing_words = ['login', 'verify', 'secure', 'update', 'account']
    features['has_login_word'] = 1 if any(word in url.lower() for word in phishing_words) else 0

    # NEW feature 2: subdomain count
    features['subdomain_count'] = max(0, url.count('.') - 1)

    # NEW feature 3: IP address in URL
    features['is_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    return features
