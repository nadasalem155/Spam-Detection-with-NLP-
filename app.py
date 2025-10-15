# spam_detection_app.py
import streamlit as st
import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="📩 Spam Detection", page_icon="🤖", layout="wide")
st.title("📩 Spam Detection ")
st.subheader("Enter any message and find out if it's 📨 Spam or 📫 Not Spam!")

# ------------------------
# Download NLTK data
# ------------------------
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------
# Load saved model & preprocessors
# ------------------------
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf.pkl")

# ------------------------
# Initialize preprocessing tools
# ------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ------------------------
# Preprocessing function
# ------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s$!]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    clean_text = " ".join(tokens)
    return clean_text

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
# ULTRA FIXED Input - NO SESSION STATE NEEDED! ✅
# ------------------------
st.text_area("✍ Enter your message here:", key="user_input", height=120)

# ------------------------
# Prediction button
# ------------------------
if st.button("Predict 🚀"):
    user_input = st.session_state.user_input  
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message to predict.")
    else:
        clean_text = preprocess_text(user_input)
        X_vec = tfidf.transform([clean_text]).toarray()
        X_scaled = scaler.transform(X_vec)
        pred_prob = mlp_model.predict_proba(X_scaled)[0]
        spam_prob = pred_prob[1] * 100
        notspam_prob = pred_prob[0] * 100
        
        if spam_prob >= 60:
            st.markdown(styled_box(
                f"🛑 Prediction: Spam<br>Confidence: {spam_prob:.2f}%",
                "#e74c3c", "🛑"
            ), unsafe_allow_html=True)
        elif notspam_prob >= 60:
            st.markdown(styled_box(
                f"✅ Prediction: Not Spam<br>Confidence: {notspam_prob:.2f}%",
                "#27ae60", "✅"
            ), unsafe_allow_html=True)
        else:
            st.markdown(styled_box(
                f"🤔 Prediction: Uncertain<br>Spam: {spam_prob:.2f}% | Not Spam: {notspam_prob:.2f}%",
                "#f39c12", "🤔"
            ), unsafe_allow_html=True)