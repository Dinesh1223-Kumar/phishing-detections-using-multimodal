from features.html_features import extract_html_features

html = """
<html>
<body>
<form>
<input type="password">
</form>
<iframe src="evil.html"></iframe>
<a href="http://evil.com">Click</a>
<p>Verify your account</p>
</body>
</html>
"""

features = extract_html_features(html)
print(features)
