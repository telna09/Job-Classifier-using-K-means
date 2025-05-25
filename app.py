
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Job Classifier", layout="wide")

st.title("💼 Intelligent Job Posting Classifier")
st.markdown("Automatically scrape, cluster, and match job listings based on your skill interests.")

st.sidebar.header("📌 App Navigation")
page = st.sidebar.radio("Go to", ["1️⃣ Scrape Jobs", "2️⃣ Cluster Jobs", "3️⃣ Match My Skills"])

# ------------------------
# Utility Functions
# ------------------------

@st.cache_data
def load_user_preferences(path='user_preferences.json'):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
    return {}

@st.cache_data
def load_jobs(file_path='raw_jobs_20250525_044013.csv'):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_resource
def load_model(model_path='models/job_classifier_20250525_044013.joblib'):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_vectorizer():
    return TfidfVectorizer(stop_words='english', max_features=500)

def scrape_karkidi_jobs():
    st.info("Scraping Karkidi.com... (dummy data loaded for this demo)")
    # Simulated data
    return pd.DataFrame({
        'title': ['Data Scientist', 'ML Engineer', 'AI Researcher'],
        'company': ['Company A', 'Company B', 'Company C'],
        'skills': ['Python, Machine Learning, SQL', 'Deep Learning, Python, NLP', 'AI, Python, Statistics']
    })

# ------------------------
# Page 1: Scrape Jobs
# ------------------------
if page == "1️⃣ Scrape Jobs":
    st.subheader("🕸️ Scrape Job Listings")
    st.markdown("Fetch the latest jobs from [Karkidi.com](https://www.karkidi.com).")

    if st.button("Start Scraping"):
        jobs_df = scrape_karkidi_jobs()
        jobs_df.to_csv("scraped_jobs.csv", index=False)
        st.success("Scraping complete. Preview below:")
        st.dataframe(jobs_df)

# ------------------------
# Page 2: Cluster Jobs
# ------------------------
elif page == "2️⃣ Cluster Jobs":
    st.subheader("🧠 Cluster Jobs Based on Skills")
    jobs_df = load_jobs()
    model = load_model()
    vectorizer = load_vectorizer()

    if jobs_df.empty or model is None:
        st.warning("Please scrape job data and ensure the trained model is available.")
    else:
        if st.button("Run Clustering"):
            skill_texts = jobs_df['skills'].fillna('')
            X = vectorizer.fit_transform(skill_texts)
            cluster_labels = model.predict(X)
            jobs_df['Cluster'] = cluster_labels
            st.success("Jobs clustered successfully.")
            st.dataframe(jobs_df[['title', 'company', 'skills', 'Cluster']])
            jobs_df.to_csv("clustered_jobs.csv", index=False)

# ------------------------
# Page 3: Match Skills
# ------------------------
elif page == "3️⃣ Match My Skills":
    st.subheader("🎯 Find Jobs Based on Your Skills")

    jobs_df = load_jobs()
    if os.path.exists("clustered_jobs.csv"):
        jobs_df = pd.read_csv("clustered_jobs.csv")

    if jobs_df.empty:
        st.warning("Please scrape and cluster job data first.")
    else:
        unique_skills = list(set(', '.join(jobs_df['skills'].dropna()).split(', ')))
        selected_skills = st.multiselect("Select your skills of interest:", sorted(unique_skills))

        if selected_skills:
            matching_jobs = jobs_df[jobs_df['skills'].str.contains('|'.join(selected_skills), case=False, na=False)]
            st.success(f"Found {len(matching_jobs)} matching job(s):")
            st.dataframe(matching_jobs[['title', 'company', 'skills']])
        else:
            st.info("Select skills to get recommendations.")
