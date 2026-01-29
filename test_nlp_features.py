from features.nlp_features import predict_nlp_from_html

phishing_html = """
<html><body>
Your account is suspended. Verify now to avoid closure.
</body></html>
"""

legit_html = """
<html><body>
Welcome to our company homepage. Learn more about our services.
</body></html>
"""

print("Phishing prediction:", predict_nlp_from_html(phishing_html))
print("Legit prediction:", predict_nlp_from_html(legit_html))
