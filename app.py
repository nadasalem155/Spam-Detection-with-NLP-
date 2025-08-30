# spam_detector_app.py
import streamlit as st
import joblib
import spacy
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="📩 Spam Detector",
    page_icon="🤖",
    layout="wide"
)
st.title("📩 Spam Detector")
st.subheader("Enter any message and find out if it's 📨 Spam or 📫 Not Spam!")

# ------------------------
# Load saved model & preprocessing objects
# ------------------------
saved_objects = joblib.load("spam_detector_full.joblib")
mlp = saved_objects["model"]
scaler = saved_objects["scaler"]
stop_words = saved_objects["stop_words"]
lemmatizer = saved_objects["lemmatizer"]

# ------------------------
# Load spaCy model here (do NOT load from joblib)
# ------------------------
nlp = spacy.load("en_core_web_md")

# ------------------------
# Download NLTK packages if not present
# ------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------
# Preprocessing function
# ------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

# ------------------------
# Styled box function
# ------------------------
def styled_box(message, color, icon):
    return f"""
    <div style="
        background-color:{color};
        padding:20px;
        border-radius:15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
        font-size:18px;
        font-weight:500;
        color:white;
        text-align:center;
    ">
        <span style="font-size:25px;">{icon}</span><br>
        {message}
    </div>
    """

# ------------------------
# Input text
# ------------------------
user_input = st.text_area("✍️ Enter your message here:")

if st.button("Predict 🚀"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to predict.")
    else:
        tokens = preprocess_text(user_input)
        clean_text = " ".join(tokens)
        vector = nlp(clean_text).vector.reshape(1, -1)
        vector_scaled = scaler.transform(vector)
        pred_prob = mlp.predict_proba(vector_scaled)[0]
        spam_prob = pred_prob[1] * 100   # probability for spam
        notspam_prob = pred_prob[0] * 100  # probability for not spam

        # ------------------------
        # Threshold logic with 3 cases
        # ------------------------
        if spam_prob >= 60:
            st.markdown(
                styled_box(
                    f"🛑 Prediction: Spam<br>"
                    f"Confidence in Spam: {spam_prob:.2f}%",
                    "#e74c3c", "🛑"
                ),
                unsafe_allow_html=True
            )
        elif notspam_prob >= 60:
            st.markdown(
                styled_box(
                    f"✅ Prediction: Not Spam<br>"
                    f"Confidence in Not Spam: {notspam_prob:.2f}%",
                    "#27ae60", "✅"
                ),
                unsafe_allow_html=True
            )
        else:
            if spam_prob >= notspam_prob:
                st.markdown(
                    styled_box(
                        f"🤔 Prediction: Uncertain (leans Spam)<br>"
                        f"Confidence in Spam: {spam_prob:.2f}%",
                        "#f39c12", "🤔"
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    styled_box(
                        f"🤔 Prediction: Uncertain (leans Not Spam)<br>"
                        f"Confidence in Not Spam: {notspam_prob:.2f}%",
                        "#f39c12", "🤔"
                    ),
                    unsafe_allow_html=True
                )