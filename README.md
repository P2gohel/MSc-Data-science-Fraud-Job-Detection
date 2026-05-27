# Detecting Fraudulent Job Postings: A Comparative Machine Learning Approach

---

## Project Overview

This project develops an automated machine learning system for detecting fraudulent job postings.
Two classifiers — Logistic Regression and Support Vector Machine (SVM) — are compared using
TF-IDF text features. The output is an interpretable fraud-likelihood score (probability),
not a binary label, so end users can make informed decisions.

## Dataset

[Real or Fake: Fake Job Posting Prediction (Kaggle, 2020)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- 17,880 job postings
- 866 fraudulent (4.84%) — heavy class imbalance
- 18 columns: text fields (title, description, requirements, benefits)
  and structured fields (has_company_logo, telecommuting, employment_type)

## Repository Structure

```
MSc-Data-science-Fraud-Job-Detection/
├── Code/                  # Notebooks 
├── Data/                  # raw dataset
├── Document/
│   └── Meeting_Log/       # meeting records (markdown)
├── Output/                # generated figures 
├── .gitignore
├── README.md
└── requirements.txt       # Python dependencies
```

## How to Reproduce

```bash
git clone https://github.com/P2gohel/MSc-Data-science-Fraud-Job-Detection.git
cd MSc-Data-science-Fraud-Job-Detection
pip install -r requirements.txt
jupyter notebook Code/EDA.ipynb
```

## Project Status

- [x] Proposal approved
- [x] Ethical approval submitted
- [x] Exploratory Data Analysis
- [ ] Literature Review chapter (in progress)
- [ ] Preprocessing pipeline
- [ ] Baseline Logistic Regression
- [ ] SVM model
- [ ] Class imbalance handling
- [ ] Evaluation & error analysis
- [ ] Prototype interface
- [ ] Dissertation report

## Methodology

1. **EDA** — explore class balance, feature distributions and text patterns
2. **Preprocessing** — text cleaning, lemmatisation, missing value handling
3. **Feature Engineering** — TF-IDF on combined text fields + structured features
4. **Modelling** — Logistic Regression vs SVM with stratified train/test split
5. **Imbalance Handling** — compare `class_weight='balanced'` and SMOTE
6. **Evaluation** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix
7. **Prototype** — interactive interface showing fraud probability and key influencing words

## Key EDA Findings

- Severe class imbalance (4.84% fraud) — accuracy is not a reliable metric
- `has_company_logo` is the strongest single signal of legitimacy
- Missing fields (`company_profile`, `requirements`, `benefits`) correlate with fraud
- Part-time and "Other" employment types have higher fraud rates than full-time
- Fake postings use distinctive vocabulary (*work from home*, *earn*, *data entry*)

