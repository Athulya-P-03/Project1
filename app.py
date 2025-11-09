# app.py
# ================================================================
#  AI-Powered Task Management System - Streamlit App
# ================================================================

import streamlit as st
import joblib
import pandas as pd
import re, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------------
#  Load Models
# -------------------------------
@st.cache_resource
def load_models():
    tfidf_priority = joblib.load("tfidf_vectorizer.pkl")
    priority_model = joblib.load("best_priority_model.pkl")
    le_priority = joblib.load("label_encoder_priority.pkl")

    tfidf_user = joblib.load("tfidf_user.pkl")
    user_model = joblib.load("user_assignment_model.pkl")
    le_user = joblib.load("label_encoder_user.pkl")

    return tfidf_priority, priority_model, le_priority, tfidf_user, user_model, le_user


tfidf_priority, priority_model, le_priority, tfidf_user, user_model, le_user = load_models()

# -------------------------------
#  NLP Preprocessing Function
# -------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens]
    return " ".join(tokens)

# -------------------------------
#  Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Task Management System", layout="centered")

st.title(" AI-Powered Task Management System")
st.write("Automatically **classifies**, **prioritizes**, and **assigns** tasks using NLP and ML.")

# Input Section
st.subheader(" Enter Task Details")
task_input = st.text_area("Describe your task:", placeholder="E.g., Prepare project report for next week")

if st.button("Predict Priority & Assignee"):
    if not task_input.strip():
        st.warning(" Please enter a task description.")
    else:
        # Preprocess input
        clean_text = preprocess_text(task_input)

        # Priority Prediction
        X_priority = tfidf_priority.transform([clean_text])
        priority_pred = priority_model.predict(X_priority)
        priority_label = le_priority.inverse_transform(priority_pred)[0]

        # User Assignment Prediction
        X_user = tfidf_user.transform([clean_text])
        user_pred = user_model.predict(X_user)
        user_label = le_user.inverse_transform(user_pred)[0]

        # Display Results
        st.success(" Prediction Complete!")
        st.markdown(f"Task: {task_input}")
        st.markdown(f"Predicted Priority: {priority_label}")
        st.markdown(f"Assigned To: {user_label}")

# Footer
st.markdown("---")
st.caption("Developed by Athulya P | AI-Powered Task Management System")

