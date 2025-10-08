import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import os
import math

st.set_page_config(page_title="AI Resume Matcher", layout="wide", page_icon="💼")

BASE_CSS = """
<style>
.block-container {padding: 1.2rem 1.5rem;}
.skill-tag {
  display:inline-block;
  padding:6px 10px;
  margin:4px 6px 4px 0;
  border-radius:12px;
  color: #fff;
  font-weight:600;
  font-size:0.9rem;
  box-shadow: 0 1px 6px rgba(0,0,0,0.25);
}
.role-card {
  padding:12px;
  border-radius:10px;
  margin-bottom:10px;
  border: 1px solid rgba(255,255,255,0.03);
}
.rt-dark .skill-tag { box-shadow: 0 1px 6px rgba(0,0,0,0.6); }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://img.icons8.com/fluency/48/000000/resume.png", width=56)
with col2:
    st.title("AI-Powered Resume–Job Matching System")
    st.write("Paste your resume text or upload a PDF. See highlighted skills and visual match scores.")

required_files = ['vectorizer.pkl', 'job_vectors.pkl', 'jobs.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"Missing files in repo root: {', '.join(missing_files)} — upload them and redeploy.")
    st.stop()

try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    job_vectors = pickle.load(open('job_vectors.pkl', 'rb'))
    jobs_df = pd.read_pickle('jobs.pkl')
except Exception as e:
    st.error(f"Error loading model/data files: {e}")
    st.stop()

skills_list = [
    'python','machine learning','sql','html','css','javascript','react',
    'java','spring boot','mysql','aws','docker','kubernetes',
    'deep learning','tensorflow','nlp','power bi','excel','data visualization'
]

synonyms = {
    'data analysis': 'data visualization',
    'data analyst': 'data visualization',
    'powerbi': 'power bi',
    'power-bi': 'power bi',
    'deep-learning': 'deep learning',
    'tensorflow': 'tensorflow',
    'pandas': 'python',
    'scikit-learn': 'machine learning',
    'sklearn': 'machine learning',
    'natural language processing': 'nlp',
    'resume': '',
    'excel': 'excel',
    'sql': 'sql'
}

skill_color_map = {
    'python': '#3572A5', 'machine learning': '#6f42c1', 'sql': '#F29111',
    'html': '#E34F26', 'css': '#264DE4', 'javascript': '#F7DF1E', 'react': '#61DAFB',
    'java': '#b07219', 'spring boot': '#6DB33F', 'mysql': '#00758F', 'aws': '#FF9900',
    'docker': '#0db7ed', 'kubernetes': '#326CE5', 'deep learning': '#ff6f61',
    'tensorflow': '#FF6F00', 'nlp': '#FF4081', 'Power BI': '#F2C811', 'Excel': '#217346',
    'data visualization': '#00A3E0'
}
default_tag_color = "#4CAF50"

def normalize_text(text: str) -> str:
    return text.lower().replace('\n',' ').replace('\r',' ').strip()

def extract_skills_smart(text: str):
    text_low = normalize_text(text)
    found = set()
    for phrase, canon in synonyms.items():
        if phrase in text_low:
            if canon and canon in skills_list:
                found.add(canon)
    for skill in skills_list:
        if skill in text_low:
            found.add(skill)
    tokens = set([t for t in text_low.replace('/', ' ').split() if len(t) > 1])
    for tok in tokens:
        if tok in synonyms and synonyms[tok] in skills_list:
            found.add(synonyms[tok])
    return sorted(found)

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        pages = []
        for pg in reader.pages:
            txt = pg.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception:
        return ""

def make_skill_tag_html(skill):
    color = skill_color_map.get(skill, default_tag_color)
    return f"<span class='skill-tag' style='background:{color};'>{skill}</span>"

def format_similarity(p):
    return f"{p*100:.2f}%"

left, right = st.columns([2,1])
with left:
    resume_text = st.text_area("Paste your resume text here:", height=180)
    uploaded_file = st.file_uploader("Or upload your PDF resume", type=["pdf"])
with right:
    st.write("Settings")
    num_jobs = st.slider("Top jobs to display", 3, 10, 5)
    dark_mode = st.checkbox("Enable dark-ish style (CSS)", value=True)
    st.write("Tip: paste a longer resume (skills + 1–2 sentences) for better matches.")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    if resume_text.strip():
        st.success("PDF uploaded and text extracted successfully!")

if dark_mode:
    DARK_CSS = """
    <style>
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .block-container { background-color: #071128; color: #e6eef8; border-radius:8px; padding:10px; }
    .st-bx { color: #fff; }
    .role-card { background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); }
    </style>
    """
    st.markdown(DARK_CSS, unsafe_allow_html=True)

if st.button("Find Matching Jobs"):
    if not resume_text or resume_text.strip() == "":
        st.warning("Please paste text or upload a PDF resume.")
    else:
        resume_norm = normalize_text(resume_text)
        extracted = extract_skills_smart(resume_text)
        st.subheader("Extracted skills")
        if extracted:
            tags_html = " ".join([make_skill_tag_html(s) for s in extracted])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("No known skills detected from the list. Try adding keywords like 'Python, SQL, Power BI'.")

        try:
            rv = vectorizer.transform([resume_text])
            sims = cosine_similarity(rv, job_vectors).flatten()
            jobs_df['similarity'] = sims
            top = jobs_df.sort_values(by='similarity', ascending=False).head(num_jobs).reset_index(drop=True)
        except Exception as e:
            st.error(f"Error computing similarity: {e}")
            top = pd.DataFrame(columns=['role','description','similarity'])

        res_col_left, res_col_right = st.columns([2, 1])
        with res_col_left:
            st.subheader(f"Top {len(top)} matches")
            for i, row in top.iterrows():
                role = row.get('role', 'Unknown role')
                desc = row.get('description', '')
                sim = float(row.get('similarity', 0.0))
                st.markdown(f"<div class='role-card'><b style='font-size:1.05rem'>{role}</b>", unsafe_allow_html=True)
                prog_label = f"Similarity: {format_similarity(sim)}"
                st.write(prog_label)
                try:
                    prog_val = int(max(0, min(100, math.floor(sim*100))))
                    st.progress(prog_val)
                except:
                    st.progress(0)
                with st.expander("Job description"):
                    st.write(desc)
                st.markdown("</div>", unsafe_allow_html=True)

        with res_col_right:
            st.subheader("Summary")
            st.metric("Top match", top.loc[0,'role'] if len(top)>0 else "—", format_similarity(top.loc[0,'similarity']) if len(top)>0 else "—")
            avg_sim = top['similarity'].mean() if len(top)>0 else 0.0
            st.metric("Average similarity", format_similarity(avg_sim))
            st.write("---")
            st.write("Skills detected:")
            if extracted:
                for s in extracted:
                    color = skill_color_map.get(s, default_tag_color)
                    st.markdown(f"<div style='display:inline-block;margin:3px;padding:6px 10px;border-radius:10px;background:{color};color:#fff;font-weight:600'>{s}</div>", unsafe_allow_html=True)
            else:
                st.write("—")
            st.write("---")
            st.write("How to improve matches:")
            st.write("- Add more details in your resume (experience, tools, keywords).")
            st.write("- Use complete phrases like 'Power BI', 'TensorFlow', 'SQL'.")
            st.write("- Upload a PDF resume for richer text.")



