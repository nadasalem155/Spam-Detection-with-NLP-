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
    # lowercase
    text = text.lower()
    # remove unwanted punctuation except $ and !
    text = re.sub(r"[^\w\s$!]", "", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    # lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    # join tokens back
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
# Input text with session state
# ------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "✍ Enter your message here:", 
    value=st.session_state.user_input, 
    key="input_area"
)

if st.button("Predict 🚀"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message to predict.")
    else:
        st.session_state.user_input = user_input  # حفظ الجملة في session
        clean_text = preprocess_text(user_input)
        X_vec = tfidf.transform([clean_text]).toarray()
        X_scaled = scaler.transform(X_vec)
        pred_prob = mlp_model.predict_proba(X_scaled)[0]
        spam_prob = pred_prob[1] * 100
        notspam_prob = pred_prob[0] * 100

        # Threshold logic
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