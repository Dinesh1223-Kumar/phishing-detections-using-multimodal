import socket
import re
from urllib.parse import urlparse

def extract_network_features(url):
    # Dictionary to store extracted network-level features
    features = {}

    # Parse the URL into components (scheme, netloc, path, etc.)
    parsed = urlparse(url)
    domain = parsed.netloc

    # Remove port number from domain if present (e.g., example.com:8080)
    domain = domain.split(":")[0]

    # 1️⃣ Length of the domain name
    features["domain_length"] = len(domain)

    # 2️⃣ Number of subdomains (e.g., a.b.example.com → 2 subdomains)
    features["num_subdomains"] = domain.count('.') - 1
    if features["num_subdomains"] < 0:
        features["num_subdomains"] = 0

    # 3️⃣ Check if domain is an IP address instead of a hostname
    features["has_ip_address"] = int(bool(
        re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)
    ))

    # 4️⃣ Check whether the domain successfully resolves via DNS
    try:
        socket.gethostbyname(domain)
        features["dns_resolves"] = 1
    except:
        features["dns_resolves"] = 0

    # 5️⃣ Check if the URL uses HTTPS protocol
    features["uses_https"] = int(url.startswith("https"))

    # Return the extracted network features
    return features
