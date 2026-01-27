from bs4 import BeautifulSoup

def extract_html_features(html):
    """
    Input:
        html (str): Raw HTML content of a webpage
    Output:
        dict: Numeric features representing the HTML
    Features included:
        - form_count: number of <form> tags
        - password_input_count: number of <input type="password">
        - iframe_count: number of <iframe> tags
        - external_link_count: number of links to external domains
        - has_suspicious_words: 1 if any suspicious words exist, else 0
        - html_length: total length of HTML string
    """

    # --- Step 1: Parse HTML safely ---
    soup = BeautifulSoup(html, "html.parser")

    # --- Step 2: Feature 1 - Number of forms ---
    form_count = len(soup.find_all("form"))

    # --- Step 3: Feature 2 - Number of password inputs ---
    password_input_count = len(soup.find_all("input", {"type": "password"}))

    # --- Step 4: Feature 3 - Number of iframes ---
    iframe_count = len(soup.find_all("iframe"))

    # --- Step 5: Feature 4 - Number of external links ---
    external_link_count = 0
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("http") and "yourdomain.com" not in href:
            external_link_count += 1
    # Note: replace "yourdomain.com" with the main domain you consider internal if needed

    # --- Step 6: Feature 5 - Suspicious keywords ---
    suspicious_words = ["login", "verify", "account", "secure", "update"]
    html_lower = html.lower()
    has_suspicious_words = 1 if any(word in html_lower for word in suspicious_words) else 0

    # --- Step 7: Feature 6 - HTML length ---
    html_length = len(html)

    # --- Step 8: Return features as a dictionary ---
    return {
        "form_count": form_count,
        "password_input_count": password_input_count,
        "iframe_count": iframe_count,
        "external_link_count": external_link_count,
        "has_suspicious_words": has_suspicious_words,
        "html_length": html_length
    }

   