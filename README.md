# 📩 Spam Detector with NLP  
An **end-to-end Natural Language Processing (NLP)** project that classifies text messages as **Spam** or **Not Spam** using multiple feature extraction techniques and ML models.  
[🔗 **Live Streamlit App**](PUT-YOUR-STREAMLIT-LINK-HERE)  

---

## **1. Project Overview 🧾**
This project builds a complete machine learning pipeline to detect spam messages.  
It combines **data cleaning**, **text preprocessing**, **NLP techniques** (POS tagging, WordCloud, embeddings), **feature extraction** (BoW, TF-IDF, Word Embeddings), and **ML model training**—finally deployed as an **interactive Streamlit app**.

---

## **2. Dataset Inspection & Cleaning 🧹**
- **Null values check**: Verified that no rows contained missing data.  
- **Duplicate removal**: Dropped duplicate rows to avoid bias.  
- **Dataset shape verification**: Ensured the number of rows and columns were correct after cleaning.

---

## **3. Text Preprocessing 🔤**
- **Lowercasing** all text to ensure uniformity.  
- **Removing punctuation** to focus on words only.  
- **Tokenizing** messages into individual words.  
- **Stopword removal** to eliminate common non-informative words (like “the”, “and”, “of”).  
- **Lemmatization** to reduce words to their base form (e.g., "running" → "run").  

This created a **clean_text column** ready for NLP processing.  

---

## **4. NLP Exploratory Analysis 🧠**

### **Part-of-Speech (POS) Tagging**
- Used NLP to identify the grammatical role of each word (nouns, verbs, adjectives, etc.).  
- Added a new column showing POS tags for every message.  
- Helped analyze which types of words frequently appear in spam vs. normal messages.  

---

### **WordCloud Visualization ☁️**
- Combined all messages into one text block.  
- Generated a WordCloud to highlight the most frequent words visually.  
- Larger words indicate higher frequency.  

![WordCloud](PUT-YOUR-WORDCLOUD-IMAGE-LINK-HERE)

This step gave quick insights into common spam keywords.

---

## **5. Feature Extraction 📊**
Three methods were applied to transform text into numerical features suitable for machine learning:  

- **Bag of Words (BoW)**: Counts word occurrences across all messages.  
- **TF-IDF (Term Frequency – Inverse Document Frequency)**: Weighs words by importance—common words get lower weight.  
- **Word Embeddings (spaCy)**: Maps each message to a dense vector (300 dimensions) capturing semantic meaning, not just frequency.  

These representations were later compared during model training.

---

## **6. Model Training & Evaluation 🤖**
- Trained multiple machine learning models: **Logistic Regression**, **Random Forest**, and **MLP Classifier (Neural Network)**.  
- Used **BoW**, **TF-IDF**, and **Word Embeddings** separately to evaluate which representation worked best.  
- Compared performance using accuracy, precision, recall, and F1-score.  

### **Model Performance Summary**
| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 96.2%    | 95.8%     | 96.5%  | 96.1%    |
| Random Forest        | 97.0%    | 96.9%     | 97.1%  | 97.0%    |
| MLP Classifier (NN)  | 98.1%    | 97.9%     | 98.2%  | 98.0%    |

> **Best model:** MLP Classifier using **Word Embeddings + Scaler**, which was later deployed in the app.

---

## **7. Streamlit App Deployment 🚀**

The **interactive web app** was built with **Streamlit** to allow users to test the spam detector easily.  

### **Features of the App**
- **User input text box**: Type or paste any message.  
- **Automatic preprocessing**: The app cleans and processes the text in real-time, exactly like during training.  
- **Real-time prediction**: Displays whether the message is *Spam* or *Not Spam*.  
- **Confidence score**: Shows probability percentages from the model.  
- **Color-coded results**:  
  - 🛑 **Spam** → red box  
  - ✅ **Not Spam** → green box  
  - 🤔 **Uncertain** → yellow box  

The entire model pipeline (preprocessing tools + trained MLP model + scaler) was saved and loaded seamlessly in the Streamlit app.

[🔗 **Try the App Here**](PUT-YOUR-STREAMLIT-LINK-HERE)

---

## **8. Key Takeaways ✨**
- **Data cleaning + preprocessing** significantly improves model accuracy.  
- **NLP insights** (POS tagging, WordCloud) help understand message patterns.  
- **Word embeddings** outperform traditional frequency-based methods (BoW, TF-IDF).  
- **Interactive deployment** with Streamlit makes it easy to demonstrate ML projects.  

---

## **9. How to Run Locally 🖥️**
1. Clone the repository.  
2. Install required dependencies (`pip install -r requirements.txt`).  
3. Run the Streamlit app using:  
   ```bash
   streamlit run spam_detector_app.py


---

10. Project Structure 📂

├── spam_detector_notebook.ipynb   # Full NLP processing and model training
├── spam_detector_app.py           # Streamlit deployment
├── spam_detector_full.joblib      # Saved model + preprocessing pipeline
├── requirements.txt               # Required dependencies
├── README.md                      # Project documentation
└── assets/wordcloud.png           # WordCloud visualization

---