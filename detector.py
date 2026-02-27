import pickle
import re

# -----------------------------
# Load Trained Model
# -----------------------------
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -----------------------------
# Clean Text Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Email Address Analysis
# -----------------------------
def analyze_email_address(email_input):
    score = 0
    reasons = []

    suspicious_tlds = ["ru", "tk", "xyz", "top"]

    match = re.search(r"[\w\.-]+@([\w\.-]+)", email_input)

    if match:
        domain = match.group(1)
        tld = domain.split(".")[-1]

        if tld in suspicious_tlds:
            score += 0.3
            reasons.append("Suspicious top-level domain detected")

        username = email_input.split("@")[0]
        if len(username) > 12 and any(char.isdigit() for char in username):
            score += 0.2
            reasons.append("Random-looking username detected")

    return score, reasons

# -----------------------------
# Main Detection Function
# -----------------------------
def detect_email(email_text):
    reasons = []

    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    ml_probability = model.predict_proba(vectorized)[0][1]

    email_score, email_reasons = analyze_email_address(email_text)
    reasons.extend(email_reasons)

    final_score = (ml_probability * 0.7) + (email_score * 0.3)

    verdict = "PHISHING" if final_score > 0.5 else "LEGITIMATE"

    return verdict, round(final_score * 100, 2), reasons