import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
job_vectors = pickle.load(open('job_vectors.pkl', 'rb'))
jobs_df = pd.read_pickle('jobs.pkl')

st.title("AI-Powered Resume–Job Matching System")
st.write("Paste your resume text below to find the most suitable jobs for you!")


resume_text = st.text_area("Paste your resume text here:")


if st.button("Find Matching Jobs"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        
        resume_vector = vectorizer.transform([resume_text])
        
        
        similarities = cosine_similarity(resume_vector, job_vectors).flatten()
        
      
        jobs_df['similarity'] = similarities
        
        
        top_matches = jobs_df.sort_values(by='similarity', ascending=False).head(3)
        
        st.subheader("Top Matching Jobs:")
        for idx, row in top_matches.iterrows():
            st.write(f"**{row['role']}** — Similarity: {row['similarity'] * 100:.2f}%")
