from bs4 import BeautifulSoup
import re

def extract_html_features_exp(html):
    """
    Experimental HTML feature extractor
    Used ONLY for algorithm comparison experiments
    """

    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # --- Core features (same as production) ---
    form_count = len(soup.find_all("form"))
    password_input_count = len(soup.find_all("input", {"type": "password"}))
    iframe_count = len(soup.find_all("iframe"))

    external_link_count = 0
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("http"):
            external_link_count += 1

    suspicious_words = [
        "login", "verify", "account", "secure",
        "update", "bank", "signin", "password"
    ]
    html_lower = html.lower()
    has_suspicious_words = int(any(word in html_lower for word in suspicious_words))

    html_length = len(html)

    # --- Experimental features ---
    script_count = len(soup.find_all("script"))
    hidden_input_count = len(
        soup.find_all("input", {"type": "hidden"})
    )

    meta_refresh = 1 if soup.find("meta", attrs={"http-equiv": "refresh"}) else 0

    suspicious_js = 1 if re.search(r"eval\(|document\.write", html_lower) else 0

    return {
        "form_count": form_count,
        "password_input_count": password_input_count,
        "iframe_count": iframe_count,
        "external_link_count": external_link_count,
        "has_suspicious_words": has_suspicious_words,
        "html_length": html_length,
        "script_count": script_count,
        "hidden_input_count": hidden_input_count,
        "meta_refresh": meta_refresh,
        "suspicious_js": suspicious_js
    }
