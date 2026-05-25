# Detecting Fraudulent Job Postings: A Comparative Machine Learning Approach

**MSc Data Science Dissertation** | Puja Gohel | Student ID: 25020509  
Supervisor: Alireza | University of the West of England

## Project Overview
Automated detection of fraudulent job postings using Logistic Regression and SVM
classifiers with TF-IDF text features. Produces an interpretable fraud-likelihood
score (probability) rather than a binary label.

## Dataset
[Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
17,880 postings | 4.84% fraudulent (866 of 17,880)

## Repository Structure
- `Notebook/` — Jupyter notebooks (numbered in run order)
- `Data/` — raw and cleaned datasets
- `Output/` — figures used in the report
- `Models/` — trained classifiers
- `Prototype/` — Streamlit demo
- `Report/` — dissertation chapters
- `meeting/` — supervisor meeting log

## How to Reproduce
```bash
pip install -r requirements.txt
jupyter notebook Notebook/01_EDA.ipynb
