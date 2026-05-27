# Meeting 3 — 26 May 2026

**Topic:** EDA review, preprocessing pipeline plan and next steps

## What I Did (Past 5 Weeks)

- Completed full Exploratory Data Analysis on the Kaggle `fake_job_postings` dataset
- Generated 7 figures (class distribution, description length, wordclouds for real/fake, fraud rate by binary features, experience, education, top industries)
- Identified 6 key findings to guide modelling decisions
- Restructured the GitHub repository with proper folder structure
- Added `README.md` with project overview, methodology, status checklist and reproduction steps
- Created `requirements.txt` for reproducibility
- Set up meeting log

## What We Discussed

- Walked through the planned **preprocessing pipeline** (text cleaning, missing value handling, feature engineering)
- Reviewed the **repository structure** — folder organisation and naming conventions
- Supervisor recommended keeping the **proposal and ethics form private** (removed from public repo)
- Supervisor recommended **prioritising the Literature Review chapter** before moving deeper into modelling, as a strong lit review will guide the methodology and benefit the rest of the report

## What I Will Do (Next 4 Weeks)

**Priority — based on supervisor recommendation:**
- Complete the **Literature Review chapter** (Chapter 2)
  - Build on the proposal's lit review with deeper critical analysis
  - Add 5–8 recent references (2022–2025)
  - Build a comparison table of existing approaches
  - Clearly identify the research gap this project fills

**Technical work (in parallel):**
- Build preprocessing pipeline (text cleaning, lemmatisation, missing value handling)
- Implement baseline Logistic Regression with TF-IDF and `class_weight='balanced'`
- Implement SVM model with calibrated probabilities

**Repository housekeeping (immediate):**
- Make repository private and add supervisor as collaborator
- Remove proposal and ethics form from public repo (keep locally)

## Key Findings Shared

1. **Severe class imbalance** — 4.84% fraud (866 of 17,880)
2. **`has_company_logo`** is the strongest single signal of legitimacy
3. **Missingness itself is informative** — `has_X` flags will capture this
4. **Employment type matters** — Part-time 9.3% vs Contract 2.9% fraud rate
5. **Distinct vocabulary** — fake: *work from home, earn, data entry* / real: *team, skill, experience*
6. **Geographic skew** — US dominates absolute count; smaller markets higher % rate by proportion