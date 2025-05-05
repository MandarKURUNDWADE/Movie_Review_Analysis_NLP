import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocess text
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().strip()
    return text

# Load the model and vectorizer
def load_model(model_choice):
    model_file = 'sentiment_model_svm.pkl' if model_choice == 'SVM' else 'sentiment_model_logistic.pkl'
    model = joblib.load(model_file)
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Predict sentiment
def predict_sentiment(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    return prediction[0], probability[0]

# Streamlit UI
st.set_page_config(page_title="IMDb Movie Review Sentiment Analysis", layout="wide")

# Title and links
st.title("🎬 IMDb Movie Review Sentiment Analysis")

col1, col2 = st.columns([8, 2])
with col2:
    st.markdown('[📁 Dataset Link]()')
    st.markdown('[💻 GitHub Link](https://github.com)')

# Problem Statement
st.write("""
## Problem Statement
The primary objective of this project is to build a machine learning classification model that can predict the sentiment of IMDb movie reviews.
The dataset contains a collection of movie reviews, and each review is labeled as either positive or negative.
""")

# Example input
st.write("### Example Review:")
st.write(""" *This movie was absolutely fantastic! The acting was superb and the storyline was captivating.*""")
st.write(""" *I hated every minute of this film. It was boring and poorly made.*""")

# Model selection
st.write("### Choose a Model for Prediction")
model_choice = st.selectbox("", ["Logistic Regression", "SVM"])

st.write("### Enter a Movie Review")
user_input = st.text_area("", "Type your review here...")

if st.button("Predict Sentiment"):
    model, vectorizer = load_model(model_choice)
    prediction, probability = predict_sentiment(user_input, model, vectorizer)
    sentiment_label = "😊 Positive" if prediction == 'positive' else "😔 Negative"
    st.write(f"### Predicted Sentiment: {sentiment_label}")
    st.write(f"### Confidence: {max(probability)*100:.2f}%")

# Model Results Table with Collapsing Section
with st.expander("Model Performance Results"):
    st.write("""
    **Model: Logistic Regression**  
    Accuracy: 0.8847  

    **Classification Report:**
    ```
                  precision    recall  f1-score   support

        negative       0.89      0.87      0.88      4961
        positive       0.88      0.90      0.89      5039

        accuracy                           0.88     10000
       macro avg       0.88      0.88      0.88     10000
    weighted avg       0.88      0.88      0.88     10000
    ```
    Confusion Matrix:
    ```
    [[4321  640]
     [ 513 4526]]
    ```

    **Model: SVM**  
    Accuracy: 0.886  

    **Classification Report:**
    ```
                  precision    recall  f1-score   support

        negative       0.89      0.88      0.88      4961
        positive       0.88      0.90      0.89      5039

        accuracy                           0.89     10000
       macro avg       0.89      0.89      0.89     10000
    weighted avg       0.89      0.89      0.89     10000
    ```
    Confusion Matrix:
    ```
    [[4341  620]
     [ 520 4519]]
    ```
    """)

# Future Work Note
st.write("""
### Future Work
I attempted to train models using BART and LSTM; however, the training process on Google Colab consistently stopped automatically after approximately 1.5 hours. Despite trying multiple times, I faced the same issue. In the future, I aim to replace the current models with higher accuracy models once the computational limitations are resolved.
""")

st.write("\nThank you for using our IMDb Movie Review Sentiment Analysis App! 🎉")
