
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Job Classifier Based on Skills", layout="wide")
st.title("📊 Job Posting Classifier")

# Load model
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Load job data
@st.cache_data
def load_job_data():
    job_files = [f for f in os.listdir() if f.startswith("raw_jobs") and f.endswith(".csv")]
    dfs = []
    for file in job_files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load user preferences
@st.cache_data
def load_user_preferences(file_path="user_preferences.json"):
    if os.path.exists(file_path):
        import json
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

# Main UI
with st.sidebar:
    st.header("🔍 User Preferences")
    skill_input = st.text_input("Enter your interested skills (comma-separated)", "python, machine learning, data analysis")
    if skill_input:
        user_skills = [s.strip().lower() for s in skill_input.split(",")]

st.subheader("📂 Classified Jobs Based on Skills")
model_path = "models/job_classifier_20250525_04..."  # Adjust the exact filename
model = load_model(model_path)
job_data = load_job_data()

if not job_data.empty and 'skills' in job_data.columns:
    job_data['skills_clean'] = job_data['skills'].fillna('').apply(lambda x: x.lower())

    # Predict cluster
    X = job_data['skills_clean']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    clusters = model.predict(X_vec)

    job_data['Cluster'] = clusters

    # Find jobs matching user skills
    def matches_user_skills(skill_text, user_skills):
        return any(skill in skill_text for skill in user_skills)

    job_data['Matched'] = job_data['skills_clean'].apply(lambda x: matches_user_skills(x, user_skills))

    # Show matched jobs
    matched_jobs = job_data[job_data['Matched']]
    st.success(f"Found {len(matched_jobs)} matching jobs based on your skills.")
    st.dataframe(matched_jobs[['title', 'company', 'skills', 'Cluster']])
else:
    st.warning("No job data available or 'skills' column missing.")
