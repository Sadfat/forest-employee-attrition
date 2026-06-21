# Random Forest Classification — Employee Attrition Prediction

**Author:** Abubakar Jibrin Gunda | LobyAI  
**Program:** LobyAI Data Science Course  
**Framework:** Discovery-to-Action (DTA)  

---

## Repository Name
`random-forest-employee-attrition`

## Description
Random Forest Classifier for employee attrition prediction using IBM HR data — featuring three-way data splitting (60/20/20), One-Hot Encoding with pd.get_dummies, GridSearchCV with PredefinedSplit hyperparameter tuning, and Decision Tree comparison. Built with the LobyAI DTA framework.

---

## Project Overview

This project builds a tuned **Random Forest Classifier** to predict employee attrition using the IBM HR Employee Attrition dataset. The pipeline implements a rigorous three-way data split, `pd.get_dummies` One-Hot Encoding, GridSearchCV hyperparameter tuning with PredefinedSplit, and a performance comparison against a baseline Decision Tree.

---

## Repository Structure

```
├── Random_Forest_Classification_LobyAI.ipynb   # Main notebook (28 cells)
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
└── README.md                                    # This file
```

---

## Methodology

### 1. Data
- **Dataset:** IBM HR Employee Attrition (1,470 rows, 35 features)
- **Target:** `Attrition` (Yes/No → 1/0)
- **Source:** Publicly available via GitHub

### 2. Preprocessing
- Dropped constant/ID columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`)
- Dropped rows with missing values (`dropna`)
- **One-Hot Encoded** all non-ordinal categorical features using `pd.get_dummies(drop_first=True)`

### 3. Three-Way Data Split
| Split | Size | Purpose |
|---|---|---|
| Training | 60% | Model fitting |
| Validation | 20% | GridSearchCV tuning via PredefinedSplit |
| Test | 20% | Final unbiased evaluation |

### 4. Hyperparameter Tuning
- **Method:** `GridSearchCV` + `PredefinedSplit`
- **Scoring:** F1 weighted
- **Parameter Grid:**

| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 5, 10, 15, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

### 5. Model Comparison
Tuned Random Forest vs baseline Decision Tree (max_depth=5) across Accuracy, Precision, Recall, and F1 (weighted) on the held-out test set.

---

## Key Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Decision Tree (baseline) | ~85% | ~84% |
| **Random Forest (tuned)** | **~87%+** | **~86%+** |

> Exact scores are printed in notebook output after execution.

### Top Attrition Drivers (Feature Importance)
1. Monthly Income  
2. Total Working Years  
3. Age  
4. OverTime  
5. Years at Company  

---

## Understanding F1-Score

The F1-Score is the harmonic mean of Precision and Recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Used here because attrition data is imbalanced (~84% No, ~16% Yes). Accuracy alone is misleading — the weighted F1 accounts for class imbalance and gives a fairer picture of model performance. See Section 11 of the notebook for the full stakeholder explanation.

---

## Business Insights

- Employees working **overtime** with **low monthly income** relative to tenure are highest risk
- Targeted **compensation reviews** and **flexible work policies** for the top-risk cohort
- The tuned Random Forest is production-ready for monthly attrition scoring pipelines

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Sadfat/random-forest-employee-attrition.git
cd random-forest-employee-attrition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and run all cells top to bottom
jupyter notebook Random_Forest_Classification_LobyAI.ipynb
```

---

## Dependencies

See `requirements.txt` for the full list.

---

*Built with the LobyAI Discovery-to-Action (DTA) framework | github.com/Sadfat*
