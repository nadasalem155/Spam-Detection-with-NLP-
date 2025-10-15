# 📩 Spam Detector with NLP  
A complete NLP pipeline that detects spam messages using advanced text preprocessing, TF-IDF feature extraction, and a Neural Network (MLP Classifier). The project includes model training, evaluation, and an interactive Streamlit web app for real-time spam detection.

[🔗 **Live Streamlit App**](https://spam-detector-with-nlp.streamlit.app/)  

---

## **1. Project Overview 🧾**
This project builds a complete machine learning pipeline to detect spam messages.  
It combines **data cleaning**, **text preprocessing**, **NLP techniques** (POS tagging, WordCloud), **feature extraction** (TF-IDF), and **ML model training**—finally deployed as an **interactive Streamlit app**.

---

## **2. Dataset Inspection & Cleaning 🧹**
- **Loaded dataset:** `spam.csv` (UCI SMS Spam Collection Dataset).  
- **Encoding fixed** using `latin1` to avoid Unicode errors.  
- **Kept only** the first two columns: `label` (ham/spam) and `text`.  
- **Renamed columns** for clarity.  
- **Removed duplicates** (403 found).  
- **Checked for nulls** – none found.  
- Final dataset shape: **(5169, 2)**.  

---

## **3. NLP Preprocessing 🔤**
Performed several text normalization steps to prepare the data for model training:

- **Lowercasing** all messages.  
- **Removing punctuation** (except `$` and `!`, since they may indicate spam).  
- **Tokenization** using `nltk.word_tokenize`.  
- **Stopword removal** using NLTK’s English stopword list.  
- **Lemmatization** using `WordNetLemmatizer`.  
- **POS Tagging** added for grammatical insight.

After processing, tokens were **joined back** into a clean text string for vectorization.

---

## **4. NLP Feature Extraction & Analysis 🧠**

### **Part-of-Speech (POS) Tagging**
Each word in every message was tagged with its part of speech (noun, verb, adjective, etc.),  
helping understand what types of words commonly appear in spam messages.  

---

### **WordCloud Visualization ☁️**
Generated a WordCloud to visualize the most frequent words in the dataset.

This highlighted common spam-related words like “free”, “win”, “call”, etc.

### 5. Feature Extraction 📊
Used TF-IDF (Term Frequency – Inverse Document Frequency) to convert text into numerical features:

max_features = 5000

Shape: (5169, 5000)

This representation weighs important words more and downplays common ones.

### 6. Handling Imbalance with SMOTE ⚖️
The dataset was imbalanced (ham: 4516, spam: 653).
Used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic spam samples and balance the dataset.

After SMOTE:
Ham: 4516  
Spam: 4516

### 7. Data Scaling ⚙️
Applied StandardScaler to normalize TF-IDF vectors for better neural network training stability.

### 8. Train-Test Split 🧪
Split the balanced dataset into:

80% training (7225 samples)

20% testing (1807 samples)

Stratified sampling to preserve class balance.

### 9. Model Training & Evaluation 🤖
Trained multiple models and compared performance:

Model	Train Accuracy	Test Accuracy
Decision Tree	96.37%	95.19%
Random Forest	93.85%	94.13%
XGBoost	99.63%	98.67%
Logistic Regression	99.75%	98.84%
SVM (Linear)	99.89%	94.69%
Perceptron	99.99%	96.07%
MLP Classifier (Neural Network)	99.75%	98.89% ✅

The MLP Classifier achieved the highest accuracy (98.9%) with strong generalization on unseen data.

MLP Classifier Report
markdown
Copy code
              precision    recall  f1-score   support
           0       1.00      0.98      0.99       904
           1       0.98      1.00      0.99       903
    accuracy                           0.99      1807
Model saved as: mlp_model.pkl

TF-IDF vectorizer saved as: tfidf.pkl

Scaler saved as: scaler.pkl

### 10. Streamlit App Deployment 🚀
The trained model was deployed using Streamlit for an interactive spam detection web app.

App Features
Clean, simple UI with icons and color-coded boxes.

Preprocessing pipeline identical to training (cleaning, tokenizing, lemmatization).

Real-time prediction with confidence scores.

Dynamic color-coded results:

🛑 Spam (red)

✅ Not Spam (green)

🤔 Uncertain (yellow)

Example Output
Input Message	Prediction	Confidence
“You won $1000 cash!”	🛑 Spam	98.7%
“Hey, are you coming tomorrow?”	✅ Not Spam	96.3%

App Code Overview (spam_detection_app.py)
Loads pre-trained mlp_model.pkl, tfidf.pkl, and scaler.pkl.

Preprocesses user input (lowercase → clean → tokenize → remove stopwords → lemmatize).

Transforms input using TF-IDF and StandardScaler.

Predicts spam probability and displays result with styled HTML box.

### 11. Key Takeaways ✨
Text preprocessing drastically improves classification accuracy.

TF-IDF + Neural Network (MLP) performs better than simpler models.

SMOTE successfully handled data imbalance.

Streamlit provides an easy and elegant way to deploy NLP apps.

### 12. How to Run Locally 🖥️
Clone this repository:

git clone https://github.com/yourusername/spam-detector-nlp.git
cd spam-detector-nlp

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run spam_detection_app.py

### 13. Project Structure 📂

├── project
    ├── project.ipynb               # Full preprocessing, training, and evaluation
    ├── spam.csv                    # Dataset 
├── app.py                          # Streamlit deployment
├── mlp_model.pkl                   # Saved trained MLP model
├── tfidf.pkl                       # Saved TF-IDF vectorizer
├── scaler.pkl                      # Saved StandardScaler
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── word-cloud.jpg                  # WordCloud visualization
### 14. Technologies Used 🧰

Python

Pandas, NumPy, Matplotlib, Seaborn

NLTK, WordCloud

scikit-learn, imblearn, xgboost

Streamlit (for deployment)

Joblib (for model serialization)

### 15. Results Summary 🏁
Best model: MLP Classifier (Neural Network)

Accuracy: 98.9%

Balanced dataset: via SMOTE

Deployment: Streamlit Web App ✅

