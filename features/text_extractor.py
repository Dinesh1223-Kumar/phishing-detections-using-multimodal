from bs4 import BeautifulSoup
import re

def extract_visible_text(html):
    """
    Extracts only human-visible text from raw HTML.
    Removes scripts, styles, and unnecessary whitespace.
    """

    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-visible / non-useful tags
    # script  -> JavaScript code
    # style   -> CSS styles
    # noscript-> Fallback content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract all remaining visible text
    # separator=" " ensures words don’t stick together
    text = soup.get_text(separator=" ")

    # Normalize whitespace:
    # - Replace multiple spaces/newlines/tabs with a single space
    # - Strip leading and trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

