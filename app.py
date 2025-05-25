
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Job Classifier", layout="wide")

st.title("🔍 Job Posting Classification Based on Required Skills")
st.markdown("This app clusters job listings from [Karkidi.com](https://www.karkidi.com) based on required skills and alerts users based on their skill interests.")

# Load user preferences
@st.cache_data
def load_user_preferences(path='user_preferences.json'):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
    return {}

# Load job data
@st.cache_data
def load_jobs(file_path='raw_jobs_20250525_044013.csv'):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# Load trained model
@st.cache_resource
def load_model(model_path='models/job_classifier_20250525_044013.joblib'):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load vectorizer
@st.cache_resource
def load_vectorizer():
    return TfidfVectorizer(stop_words='english', max_features=500)

# Main logic
jobs_df = load_jobs()
model = load_model()
vectorizer = load_vectorizer()
preferences = load_user_preferences()

if jobs_df.empty or model is None:
    st.warning("Missing job data or trained model. Please ensure required files are present.")
else:
    st.subheader("1. Preview of Scraped Jobs")
    st.dataframe(jobs_df[['title', 'company', 'skills']].head(10))

    st.subheader("2. Select Your Skills of Interest")
    unique_skills = list(set(', '.join(jobs_df['skills'].dropna()).split(', ')))
    selected_skills = st.multiselect("Choose your preferred skills:", sorted(unique_skills))

    if selected_skills:
        st.subheader("3. Job Alerts Based on Your Skills")
        matching_jobs = jobs_df[jobs_df['skills'].str.contains('|'.join(selected_skills), case=False, na=False)]
        if not matching_jobs.empty:
            st.success(f"Found {len(matching_jobs)} job(s) matching your interests.")
            st.dataframe(matching_jobs[['title', 'company', 'skills']])
        else:
            st.info("No matching jobs found today.")

    st.subheader("4. Cluster Jobs Based on Required Skills")
    if st.button("Run Clustering"):
        skill_texts = jobs_df['skills'].fillna('')
        X = vectorizer.fit_transform(skill_texts)
        cluster_labels = model.predict(X)
        jobs_df['Cluster'] = cluster_labels
        st.success("Clustering complete. Preview below:")
        st.dataframe(jobs_df[['title', 'company', 'skills', 'Cluster']].head(10))
