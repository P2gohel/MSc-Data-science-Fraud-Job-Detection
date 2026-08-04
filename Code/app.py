import streamlit as st

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🔍")

import joblib
import numpy as np
import pandas as pd
import re
import shap
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

#Force wordnet to fully load before the lemmatizer is used
from nltk.corpus import wordnet
wordnet.ensure_loaded()


#Load the saved model once and keep it in memory
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../Model/lr_model.pkl')
        tfidf = joblib.load('../Model/tfidf_vectorizer.pkl')
        feature_names = joblib.load('../Model/feature_names.pkl')
        return model, tfidf, feature_names
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model, tfidf, feature_names = load_model()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Same cleaning function used in training
def clean_text(text):
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [ lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

st.title("Fake Job Posting Detector")
st.caption("Paste a job posting to check how likely it is to be fraudulent")

job_text = st.text_area("Job posting text", height=180,
                        placeholder="Paste the job description here...")

st.write("Does the posting include:")
col1, col2 = st.columns(2)
with col1:
    has_logo = st.checkbox("Company logo")
    has_profile = st.checkbox("Company profile")
    has_questions = st.checkbox("Screening questions")
with col2:
    has_reqs = st.checkbox("Requirements section")
    has_benefits = st.checkbox("Benefits section")
    telecommuting = st.checkbox("Remote / work from home")


if st.button("Check posting", type="primary"):
    if not job_text.strip():
        st.warning("Please paste a job posting first.")
    else:
        cleaned = clean_text(job_text)
        x_tfidf = tfidf.transform([cleaned])
        x_meta = csr_matrix([[int(telecommuting), int(has_logo), int(has_questions),
                              int(has_profile), int(has_reqs), int(has_benefits)]])
        x = hstack([x_tfidf, x_meta]).tocsr()

        score = model.predict_proba(x)[0, 1]

        #Shap explanation for this posting
        explainer = shap.LinearExplainer(model, x)
        sv = explainer.shap_values(x)[0]
        present = x.toarray().flatten() != 0
        contrib = pd.DataFrame({
            'feature': np.array(feature_names)[present],
            'shap': sv[present]
        })

        #Show the score
        pct = score * 100
        if score >= 0.6:
            st.error(f"### {pct:.1f}% - Likely fraudulent")
        elif score >= 0.4:
            st.warning(f"### {pct:.1f}% - Uncertain, review carefully")
        else:
            st.success(f"### {pct:.1f}% - Likely legitimate")


        st.progress(float(score))

        st.subheader("Why?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Signs of fraud**")
            fraud_words = contrib[contrib['shap'] > 0].nlargest(8, 'shap')
            if len(fraud_words):
                for _, r in fraud_words.iterrows():
                    st.markdown(f" 🔴 {r['feature']}")
            else:
                st.caption("None detected")
        with c2:
            st.markdown("**Signs of legitimacy**")
            real_words = contrib[contrib['shap'] < 0].nsmallest(8, 'shap')
            if len(real_words):
                for _, r in real_words.iterrows():
                    st.markdown(f" 🟢 {r['feature']}")
            else:
                st.caption("None Detected")

        st.caption("Model: Logistic Regression with class weighting | "
                   "Fraud recall 90.8% | AUC-ROC 0.989")