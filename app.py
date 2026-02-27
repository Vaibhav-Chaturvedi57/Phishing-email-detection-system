import streamlit as st
from detector import detect_email

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="🛡",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("🛡 Phishing Email Detection System")
st.markdown("### Hybrid ML + Rule-Based Detection Engine")

st.write(
    "This system analyzes email content using a trained Machine Learning model "
    "combined with rule-based domain analysis."
)

st.divider()

# -----------------------------
# User Input
# -----------------------------
email_text = st.text_area(
    "📨 Paste Email Content Here:",
    height=220,
    placeholder="Example: Your account has been suspended. Click here to verify immediately."
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter email content before analyzing.")
    else:
        verdict, confidence, reasons = detect_email(email_text)

        st.divider()

        # Display Result
        if verdict == "PHISHING":
            st.error(f"⚠ Phishing Detected ({confidence}%)")
        else:
            st.success(f"✅ Legitimate Email ({confidence}%)")

        # Confidence Bar
        st.progress(confidence / 100)

        # Risk Indicators
        st.subheader("🔍 Risk Indicators")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("No strong risk indicators detected.")

st.divider()

st.caption("Built using Scikit-Learn, TF-IDF, and Streamlit")