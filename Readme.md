# MLE Case Study 1 — Hospital Readmission Prediction

A machine learning case study that predicts whether a patient will be readmitted to hospital, using clinical and demographic data from ~25,000 patient encounters.

## Problem Statement

Hospital readmissions are costly and often indicate gaps in patient care. This project builds a binary classification model to predict the `readmitted` outcome (`yes` / `no`) for a patient based on their hospital stay details, diagnoses, and treatment.

## Dataset

`content/hospital_readmissions.csv` — 25,000 rows, 17 columns:

| Column | Description |
|---|---|
| `age` | Age bracket (e.g. `[70-80)`) |
| `time_in_hospital` | Days spent in hospital |
| `n_lab_procedures` | Number of lab procedures performed |
| `n_procedures` | Number of procedures performed |
| `n_medications` | Number of medications administered |
| `n_outpatient` | Outpatient visits in the prior year |
| `n_inpatient` | Inpatient visits in the prior year |
| `n_emergency` | Emergency visits in the prior year |
| `medical_specialty` | Specialty of the admitting physician |
| `diag_1`, `diag_2`, `diag_3` | Primary/secondary/tertiary diagnosis categories |
| `glucose_test` | Result of glucose test (`no`/`normal`/`high`) |
| `A1Ctest` | Result of A1C test |
| `change` | Whether diabetic medication was changed |
| `diabetes_med` | Whether a diabetes medication was prescribed |
| `readmitted` | **Target** — whether the patient was readmitted |

No missing values or duplicate rows were found in the dataset.

## Approach

1. **EDA** — checked shape, dtypes, nulls, and duplicates (`Case_Study_1_Hospital_Readmission.ipynb`).
2. **Preprocessing**
   - Encoded the target (`yes` → `1`, `no` → `0`).
   - One-hot encoded categorical features (`age`, `medical_specialty`, `diag_1/2/3`, `glucose_test`, `A1Ctest`, `change`, `diabetes_med`).
   - Split into train/test sets (80/20, stratified on the target).
   - Standardized numeric features with `StandardScaler`.
3. **Modeling** — trained a `LogisticRegression` classifier (L2 penalty, `liblinear` solver).
4. **Evaluation** — assessed performance with ROC-AUC, ROC curve, and confusion matrix.

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.6448 |

## Project Structure

```
MLE_CASE_STUDY_1/
├── Readme.md
├── Case_Study_1_Hospital_Readmission.ipynb   # Main analysis & model notebook
└── content/
    └── hospital_readmissions.csv             # Dataset
```

## Getting Started

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook Case_Study_1_Hospital_Readmission.ipynb
```

Update the CSV path in the notebook's first cell (`pd.read_csv(...)`) to point to `content/hospital_readmissions.csv` if not running on Google Colab.
