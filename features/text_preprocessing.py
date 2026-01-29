import re

def clean_text(text):
    """
    Preprocess raw text for NLP phishing detection.
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove special characters and numbers
    4. Normalize whitespace
    """

    # 1️⃣ Convert all text to lowercase (case normalization)
    text = text.lower()

    # 2️⃣ Remove URLs (http, https, www links)
    # \S+  -> matches everything until a space
    text = re.sub(r"http\S+", "", text)

    # 3️⃣ Remove all characters except lowercase letters and spaces
    # [^a-z\s] -> anything that is NOT a letter or space
    text = re.sub(r"[^a-z\s]", "", text)

    # 4️⃣ Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # 5️⃣ Remove leading and trailing whitespace
    return text.strip()
