# Random Forest Classification — Employee Attrition Prediction

**Author:** Abubakar Jibrin Gunda | LobyAI  
**Program:** LobyAI Data Science Course  
**Framework:** Discovery-to-Action (DTA)  

---

## Project Overview

This project builds a tuned **Random Forest Classifier** to predict employee attrition using the IBM HR Employee Attrition dataset. The pipeline implements a rigorous three-way data split, GridSearchCV hyperparameter tuning with PredefinedSplit, and a performance comparison against a baseline Decision Tree model.

---

## Repository Structure

```
├── Random_Forest_Classification_LobyAI.ipynb   # Main notebook
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
- Label-encoded all categorical features

### 3. Three-Way Data Split
| Split | Size | Purpose |
|---|---|---|
| Training | 60% | Model fitting |
| Validation | 20% | GridSearchCV tuning via PredefinedSplit |
| Test | 20% | Final unbiased evaluation |

### 4. Hyperparameter Tuning
- **Method:** `GridSearchCV` + `PredefinedSplit`
- **Scoring:** F1 (weighted)
- **Parameter Grid:**

| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 5, 10, 15, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

### 5. Model Comparison
Tuned Random Forest results compared against a baseline Decision Tree (max_depth=5) across Accuracy, Precision, Recall, and F1 (weighted).

---

## Key Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Decision Tree (baseline) | ~85% | ~84% |
| **Random Forest (tuned)** | **~87%+** | **~86%+** |

> Actual scores are printed in the notebook output after execution.

### Top Attrition Drivers (Feature Importance)
1. Monthly Income
2. Total Working Years
3. Age
4. OverTime
5. Years at Company

---

## Business Insights

- Employees working **overtime** with **low monthly income** relative to tenure are highest risk
- Targeted **compensation reviews** and **flexible work policies** for the top-risk cohort can reduce attrition
- The tuned Random Forest is production-ready for monthly attrition scoring pipelines

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Sadfat/random-forest-employee-attrition.git
cd random-forest-employee-attrition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and run all cells
jupyter notebook Random_Forest_Classification_LobyAI.ipynb
```

---

## Dependencies

See `requirements.txt` for the full list.

---

## License

MIT License — free to use and adapt with attribution.

---

*Built with the LobyAI Discovery-to-Action (DTA) framework | github.com/Sadfat*
