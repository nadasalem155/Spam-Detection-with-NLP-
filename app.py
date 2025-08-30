import streamlit as st
import pickle
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
with open("spam_detector_full.pkl", "rb") as f:
    saved_objects = pickle.load(f)

mlp = saved_objects["model"]
scaler = saved_objects["scaler"]
stop_words = saved_objects["stop_words"]
lemmatizer = saved_objects["lemmatizer"]

# ------------------------
# Load spaCy model here (do NOT load from pickle)
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
        pred = mlp.predict(vector_scaled)[0]
        pred_prob = mlp.predict_proba(vector_scaled)[0]

        if pred == 1:
            st.markdown(
                f"<div style='border:2px solid red; padding:15px; border-radius:10px;'>"
                f"🛑 This message is predicted as Spam!<br>"
                f"Confidence: {pred_prob[1]*100:.2f}%"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='border:2px solid green; padding:15px; border-radius:10px;'>"
                f"✅ This message is predicted as Not Spam!<br>"
                f"Confidence: {pred_prob[0]*100:.2f}%"
                f"</div>", unsafe_allow_html=True
            )