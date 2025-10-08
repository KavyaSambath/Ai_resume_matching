import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import os

st.set_page_config(page_title="AI Resume Matcher", layout="wide", page_icon="💼")
st.title(" AI-Powered Resume–Job Matching System")
st.write("Paste your resume text or upload a PDF to find the most suitable jobs!")

required_files = ['vectorizer.pkl', 'job_vectors.pkl', 'jobs.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f" Missing files: {', '.join(missing_files)}. Please upload them in the app folder.")
    st.stop()

try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    job_vectors = pickle.load(open('job_vectors.pkl', 'rb'))
    jobs_df = pd.read_pickle('jobs.pkl')
except Exception as e:
    st.error(f" Error loading files: {e}")
    st.stop()

skills_list = [
    'python', 'machine learning', 'sql', 'html', 'css', 'javascript', 'react',
    'java', 'spring boot', 'mysql', 'aws', 'docker', 'kubernetes',
    'deep learning', 'tensorflow', 'nlp', 'power bi', 'excel', 'data visualization'
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

def extract_text_from_pdf(pdf_file):
    try:
        pdf = PdfReader(pdf_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except:
        return ""

resume_text = st.text_area(" Paste your resume text here:")
uploaded_file = st.file_uploader(" Or upload your PDF resume", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success(" PDF uploaded and text extracted successfully!")

num_jobs = st.slider("Select number of top jobs to display:", min_value=3, max_value=10, value=5)

if st.button("Find Matching Jobs"):
    if resume_text.strip() == "":
        st.warning(" Please enter or upload a resume!")
    else:
        extracted_skills = extract_skills(resume_text)
        st.subheader(" Extracted Skills from Resume:")
        if extracted_skills:
            skill_tags = " ".join([
                f"<span style='color: white; background-color: #4CAF50; padding:4px; margin:2px; border-radius:5px;'>{skill}</span>" 
                for skill in extracted_skills
            ])
            st.markdown(skill_tags, unsafe_allow_html=True)
        else:
            st.write("No recognized skills found.")

        try:
            resume_vector = vectorizer.transform([resume_text])
            similarities = cosine_similarity(resume_vector, job_vectors).flatten()
            jobs_df['similarity'] = similarities

            top_matches = jobs_df.sort_values(by='similarity', ascending=False).head(num_jobs)
            st.subheader(f" Top {num_jobs} Matching Jobs:")
            display_df = top_matches[['role', 'similarity']].copy()
            display_df['similarity'] = display_df['similarity'] * 100
            display_df = display_df.rename(columns={'role': 'Job Role', 'similarity': 'Similarity (%)'})
            st.dataframe(display_df.style.format({"Similarity (%)": "{:.2f}"}))
        except Exception as e:
            st.error(f" Error processing resume: {e}")

