from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_behavioral_features(html, page_url=""):
    """
    Extract behavioral / heuristic phishing indicators from HTML.
    Returns a dictionary of numeric features.
    """

    features = {}

    soup = BeautifulSoup(html, "html.parser")

    # 1️⃣ Login form present
    forms = soup.find_all("form")
    features["has_login_form"] = 1 if len(forms) > 0 else 0

    # 2️⃣ Password fields
    password_inputs = soup.find_all("input", {"type": "password"})
    features["password_input_count"] = len(password_inputs)

    # 3️⃣ Submit buttons
    submit_buttons = soup.find_all("input", {"type": "submit"}) + soup.find_all("button", {"type": "submit"})
    features["has_submit_button"] = 1 if len(submit_buttons) > 0 else 0

    # 4️⃣ Hidden inputs
    hidden_inputs = soup.find_all("input", {"type": "hidden"})
    features["hidden_input_count"] = len(hidden_inputs)

    # 5️⃣ Urgent / phishing words
    urgent_words = [
        "verify", "urgent", "immediately",
        "suspended", "confirm", "security",
        "update", "login"
    ]
    text = soup.get_text(" ").lower()
    features["has_urgent_words"] = 1 if any(word in text for word in urgent_words) else 0

    # 6️⃣ Meta refresh (auto redirect)
    meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
    features["has_meta_refresh"] = 1 if meta_refresh else 0

    # 7️⃣ External form submission
    external_action = 0
    if page_url:
        page_domain = urlparse(page_url).netloc
        for form in forms:
            action = form.get("action", "")
            if action.startswith("http"):
                action_domain = urlparse(action).netloc
                if action_domain and action_domain != page_domain:
                    external_action = 1
                    break

    features["form_action_external"] = external_action

    return features
