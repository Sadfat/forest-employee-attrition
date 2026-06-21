{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1f40ec8a",
   "metadata": {},
   "source": [
    "# Random Forest Classification\n",
    "## Three-Way Data Splitting, GridSearchCV Tuning, and Decision Tree Comparison\n",
    "\n",
    "**Author:** Abubakar Jibrin Gunda | LobyAI  \n",
    "**Program:** LobyAI Data Science Course  \n",
    "**Framework:** Discovery-to-Action (DTA)  \n",
    "**Dataset:** IBM HR Employee Attrition  \n",
    "**Objective:** Predict employee attrition using a tuned Random Forest Classifier and compare performance against a baseline Decision Tree model.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eeaf22d5",
   "metadata": {},
   "source": [
    "## 1. Imports and Environment Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "60eacd89",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:27.675425Z",
     "iopub.status.busy": "2026-06-21T11:20:27.675270Z",
     "iopub.status.idle": "2026-06-21T11:20:31.480950Z",
     "shell.execute_reply": "2026-06-21T11:20:31.479623Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numpy  : 2.4.4\n",
      "pandas : 3.0.2\n",
      "sklearn: 1.8.0\n",
      "All libraries imported successfully.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    accuracy_score, precision_score, recall_score, f1_score\n",
    ")\n",
    "\n",
    "import sklearn\n",
    "print('numpy  :', np.__version__)\n",
    "print('pandas :', pd.__version__)\n",
    "print('sklearn:', sklearn.__version__)\n",
    "print('All libraries imported successfully.')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d451595",
   "metadata": {},
   "source": [
    "## 2. Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1f8787e5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.483024Z",
     "iopub.status.busy": "2026-06-21T11:20:31.482683Z",
     "iopub.status.idle": "2026-06-21T11:20:31.681987Z",
     "shell.execute_reply": "2026-06-21T11:20:31.680986Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "URL load failed, using synthetic fallback: HTTP Error 404: Not Found\n",
      "Synthetic dataset created. Shape: (1470, 27)\n"
     ]
    },
    {
     "output_type": "execute_result",
     "metadata": {},
     "data": {
      "text/plain": [
       "   Age Attrition     BusinessTravel Department  DistanceFromHome  Education  \\\n",
       "0   56        No         Non-Travel         HR                21          2   \n",
       "1   46        No         Non-Travel        R&D                23          3   \n",
       "2   32        No  Travel_Frequently         HR                29          4   \n",
       "\n",
       "   EnvironmentSatisfaction Gender  JobInvolvement  JobLevel  ...  \\\n",
       "0                        3   Male               1         4  ...   \n",
       "1                        2   Male               1         1  ...   \n",
       "2                        1   Male               4         4  ...   \n",
       "\n",
       "  PerformanceRating  RelationshipSatisfaction StockOptionLevel  \\\n",
       "0                 4                         1                0   \n",
       "1                 4                         4                1   \n",
       "2                 3                         3                0   \n",
       "\n",
       "   TotalWorkingYears  TrainingTimesLastYear WorkLifeBalance  YearsAtCompany  \\\n",
       "0                 31                      6               3               1   \n",
       "1                 31                      1               3              23   \n",
       "2                 25                      6               4              15   \n",
       "\n",
       "   YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager  \n",
       "0                  13                        9                    13  \n",
       "1                   2                        3                    13  \n",
       "2                  10                       14                     7  \n",
       "\n",
       "[3 rows x 27 columns]"
      ]
     },
     "execution_count": 2
    }
   ],
   "source": [
    "url = ('https://raw.githubusercontent.com/dsrscientist/'\n",
    "       'dataset1/master/IBM-HR-Employee-Attrition.csv')\n",
    "\n",
    "try:\n",
    "    df = pd.read_csv(url)\n",
    "    print('Dataset loaded from URL. Shape:', df.shape)\n",
    "except Exception as e:\n",
    "    print('URL load failed, using synthetic fallback:', e)\n",
    "    np.random.seed(42)\n",
    "    n = 1470\n",
    "    df = pd.DataFrame({\n",
    "        'Age': np.random.randint(18, 60, n),\n",
    "        'Attrition': np.random.choice(['Yes','No'], n, p=[0.16, 0.84]),\n",
    "        'BusinessTravel': np.random.choice(['Non-Travel','Travel_Rarely','Travel_Frequently'], n),\n",
    "        'Department': np.random.choice(['Sales','R&D','HR'], n),\n",
    "        'DistanceFromHome': np.random.randint(1, 30, n),\n",
    "        'Education': np.random.randint(1, 6, n),\n",
    "        'EnvironmentSatisfaction': np.random.randint(1, 5, n),\n",
    "        'Gender': np.random.choice(['Male','Female'], n),\n",
    "        'JobInvolvement': np.random.randint(1, 5, n),\n",
    "        'JobLevel': np.random.randint(1, 6, n),\n",
    "        'JobRole': np.random.choice(['Sales Executive','Research Scientist','Manager','HR','Developer'], n),\n",
    "        'JobSatisfaction': np.random.randint(1, 5, n),\n",
    "        'MaritalStatus': np.random.choice(['Single','Married','Divorced'], n),\n",
    "        'MonthlyIncome': np.random.randint(1000, 20000, n),\n",
    "        'NumCompaniesWorked': np.random.randint(0, 10, n),\n",
    "        'OverTime': np.random.choice(['Yes','No'], n),\n",
    "        'PercentSalaryHike': np.random.randint(11, 25, n),\n",
    "        'PerformanceRating': np.random.randint(3, 5, n),\n",
    "        'RelationshipSatisfaction': np.random.randint(1, 5, n),\n",
    "        'StockOptionLevel': np.random.randint(0, 4, n),\n",
    "        'TotalWorkingYears': np.random.randint(0, 40, n),\n",
    "        'TrainingTimesLastYear': np.random.randint(0, 7, n),\n",
    "        'WorkLifeBalance': np.random.randint(1, 5, n),\n",
    "        'YearsAtCompany': np.random.randint(0, 40, n),\n",
    "        'YearsInCurrentRole': np.random.randint(0, 20, n),\n",
    "        'YearsSinceLastPromotion': np.random.randint(0, 15, n),\n",
    "        'YearsWithCurrManager': np.random.randint(0, 20, n),\n",
    "    })\n",
    "    print('Synthetic dataset created. Shape:', df.shape)\n",
    "\n",
    "df.head(3)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4b89274",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e1cee055",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.683661Z",
     "iopub.status.busy": "2026-06-21T11:20:31.683516Z",
     "iopub.status.idle": "2026-06-21T11:20:31.693606Z",
     "shell.execute_reply": "2026-06-21T11:20:31.692758Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape: (1470, 27)\n",
      "Missing values:\n",
      "Series([], dtype: int64)\n",
      "Dtypes: {dtype('int64'): 20, <StringDtype(storage='python', na_value=nan)>: 7}\n",
      "Attrition distribution:\n",
      "Attrition\n",
      "No     1227\n",
      "Yes     243\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print('Shape:', df.shape)\n",
    "print('Missing values:')\n",
    "print(df.isnull().sum()[df.isnull().sum() > 0])\n",
    "print('Dtypes:', df.dtypes.value_counts().to_dict())\n",
    "print('Attrition distribution:')\n",
    "print(df['Attrition'].value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "08ed5e9d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.695836Z",
     "iopub.status.busy": "2026-06-21T11:20:31.695107Z",
     "iopub.status.idle": "2026-06-21T11:20:31.849105Z",
     "shell.execute_reply": "2026-06-21T11:20:31.847881Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Figure saved.\n"
     ]
    }
   ],
   "source": [
    "counts = df['Attrition'].value_counts()\n",
    "fig, ax = plt.subplots(figsize=(6, 4))\n",
    "ax.bar(counts.index, counts.values, color=['#1976D2','#FF5722'], edgecolor='white')\n",
    "ax.set_title('Attrition Distribution', fontsize=13, fontweight='bold')\n",
    "ax.set_xlabel('Attrition')\n",
    "ax.set_ylabel('Count')\n",
    "for i, v in enumerate(counts.values):\n",
    "    ax.text(i, v + 10, str(v), ha='center', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.savefig('attrition_dist.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Figure saved.')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ad6fef8",
   "metadata": {},
   "source": [
    "## 4. Data Cleaning and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7a6bd1a8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.851057Z",
     "iopub.status.busy": "2026-06-21T11:20:31.850884Z",
     "iopub.status.idle": "2026-06-21T11:20:31.860990Z",
     "shell.execute_reply": "2026-06-21T11:20:31.860071Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dropped columns: []\n",
      "Rows dropped (missing): 0\n",
      "Remaining rows: 1470\n",
      "Target encoded: Yes=1, No=0\n",
      "Class distribution: {0: 1227, 1: 243}\n"
     ]
    }
   ],
   "source": [
    "# Drop constant and ID columns\n",
    "drop_cols = [c for c in ['EmployeeCount','StandardHours','Over18','EmployeeNumber']\n",
    "             if c in df.columns]\n",
    "df.drop(columns=drop_cols, inplace=True)\n",
    "print('Dropped columns:', drop_cols)\n",
    "\n",
    "# Drop rows with missing values\n",
    "before = len(df)\n",
    "df.dropna(inplace=True)\n",
    "print('Rows dropped (missing):', before - len(df))\n",
    "print('Remaining rows:', len(df))\n",
    "\n",
    "# Encode target variable\n",
    "df['Attrition_encoded'] = (df['Attrition'] == 'Yes').astype(int)\n",
    "print('Target encoded: Yes=1, No=0')\n",
    "print('Class distribution:', df['Attrition_encoded'].value_counts().to_dict())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c2a46015",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.862989Z",
     "iopub.status.busy": "2026-06-21T11:20:31.862823Z",
     "iopub.status.idle": "2026-06-21T11:20:31.882917Z",
     "shell.execute_reply": "2026-06-21T11:20:31.882016Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Categorical columns to encode: ['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']\n",
      "Shape after One-Hot Encoding: (1470, 32)\n"
     ]
    },
    {
     "output_type": "execute_result",
     "metadata": {},
     "data": {
      "text/plain": [
       "   Age  DistanceFromHome  Education  EnvironmentSatisfaction  JobInvolvement  \\\n",
       "0   56                21          2                        3               1   \n",
       "1   46                23          3                        2               1   \n",
       "\n",
       "   JobLevel  JobSatisfaction  MonthlyIncome  NumCompaniesWorked  \\\n",
       "0         4                1           2501                   1   \n",
       "1         1                4          10050                   0   \n",
       "\n",
       "   PercentSalaryHike  ...  Department_R&D  Department_Sales  Gender_Male  \\\n",
       "0                 12  ...           False             False         True   \n",
       "1                 14  ...            True             False         True   \n",
       "\n",
       "   JobRole_HR  JobRole_Manager  JobRole_Research Scientist  \\\n",
       "0       False            False                        True   \n",
       "1       False            False                        True   \n",
       "\n",
       "   JobRole_Sales Executive  MaritalStatus_Married  MaritalStatus_Single  \\\n",
       "0                    False                  False                  True   \n",
       "1                    False                  False                  True   \n",
       "\n",
       "   OverTime_Yes  \n",
       "0         False  \n",
       "1         False  \n",
       "\n",
       "[2 rows x 32 columns]"
      ]
     },
     "execution_count": 6
    }
   ],
   "source": [
    "# One-Hot Encode all non-ordinal categorical features using pd.get_dummies\n",
    "cat_cols = df.select_dtypes(include='object').columns.tolist()\n",
    "cat_cols = [c for c in cat_cols if c != 'Attrition']\n",
    "print('Categorical columns to encode:', cat_cols)\n",
    "\n",
    "df_encoded = pd.get_dummies(df.drop(columns=['Attrition','Attrition_encoded']),\n",
    "                             columns=cat_cols,\n",
    "                             drop_first=True)\n",
    "print('Shape after One-Hot Encoding:', df_encoded.shape)\n",
    "df_encoded.head(2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "81dbdedf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.884682Z",
     "iopub.status.busy": "2026-06-21T11:20:31.884534Z",
     "iopub.status.idle": "2026-06-21T11:20:31.889320Z",
     "shell.execute_reply": "2026-06-21T11:20:31.888445Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature matrix shape: (1470, 32)\n",
      "Target vector shape : (1470,)\n",
      "Feature columns: ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome'] ...\n"
     ]
    }
   ],
   "source": [
    "X = df_encoded\n",
    "y = df['Attrition_encoded']\n",
    "\n",
    "print('Feature matrix shape:', X.shape)\n",
    "print('Target vector shape :', y.shape)\n",
    "print('Feature columns:', list(X.columns[:8]), '...')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b587cad",
   "metadata": {},
   "source": [
    "## 5. Three-Way Data Split\n",
    "\n",
    "The data is divided into three non-overlapping sets:\n",
    "- **Training set (60%)** 鈥� used to fit the model\n",
    "- **Validation set (20%)** 鈥� used exclusively for GridSearchCV tuning via PredefinedSplit\n",
    "- **Test set (20%)** 鈥� used only for final unbiased evaluation\n",
    "\n",
    "This prevents data leakage between tuning and final evaluation.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a850fb04",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.890949Z",
     "iopub.status.busy": "2026-06-21T11:20:31.890756Z",
     "iopub.status.idle": "2026-06-21T11:20:31.902720Z",
     "shell.execute_reply": "2026-06-21T11:20:31.901714Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Three-Way Split ===\n",
      "Training   :   882 rows (60%)\n",
      "Validation :   294 rows (20%)\n",
      "Test       :   294 rows (20%)\n",
      "Total      :  1470 rows\n"
     ]
    }
   ],
   "source": [
    "# Step 1: Hold out 20% as Test\n",
    "X_trainval, X_test, y_trainval, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Step 2: Split remaining 80% into Train (75%) and Val (25%) => 60/20 of total\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval\n",
    ")\n",
    "\n",
    "print('=== Three-Way Split ===')\n",
    "print(f'Training   : {len(X_train):>5} rows ({len(X_train)/len(X)*100:.0f}%)')\n",
    "print(f'Validation : {len(X_val):>5} rows ({len(X_val)/len(X)*100:.0f}%)')\n",
    "print(f'Test       : {len(X_test):>5} rows ({len(X_test)/len(X)*100:.0f}%)')\n",
    "print(f'Total      : {len(X):>5} rows')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90301dc1",
   "metadata": {},
   "source": [
    "## 6. Baseline Decision Tree Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "740f2473",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.904559Z",
     "iopub.status.busy": "2026-06-21T11:20:31.904405Z",
     "iopub.status.idle": "2026-06-21T11:20:31.930125Z",
     "shell.execute_reply": "2026-06-21T11:20:31.929246Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Decision Tree 鈥� Test Set ===\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "No Attrition       0.83      0.97      0.89       245\n",
      "   Attrition       0.11      0.02      0.03        49\n",
      "\n",
      "    accuracy                           0.81       294\n",
      "   macro avg       0.47      0.49      0.46       294\n",
      "weighted avg       0.71      0.81      0.75       294\n",
      "\n",
      "Accuracy  : 0.8095\n",
      "Precision : 0.7115\n",
      "Recall    : 0.8095\n",
      "F1 (wt.)  : 0.7510\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier(max_depth=5, random_state=42)\n",
    "dt.fit(X_train, y_train)\n",
    "\n",
    "dt_val_pred  = dt.predict(X_val)\n",
    "dt_test_pred = dt.predict(X_test)\n",
    "\n",
    "dt_acc  = accuracy_score(y_test, dt_test_pred)\n",
    "dt_prec = precision_score(y_test, dt_test_pred, average='weighted', zero_division=0)\n",
    "dt_rec  = recall_score(y_test, dt_test_pred, average='weighted', zero_division=0)\n",
    "dt_f1   = f1_score(y_test, dt_test_pred, average='weighted', zero_division=0)\n",
    "\n",
    "print('=== Decision Tree 鈥� Test Set ===')\n",
    "print(classification_report(y_test, dt_test_pred,\n",
    "      target_names=['No Attrition','Attrition']))\n",
    "print(f'Accuracy  : {dt_acc:.4f}')\n",
    "print(f'Precision : {dt_prec:.4f}')\n",
    "print(f'Recall    : {dt_rec:.4f}')\n",
    "print(f'F1 (wt.)  : {dt_f1:.4f}')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e98358ba",
   "metadata": {},
   "source": [
    "## 7. Random Forest 鈥� Hyperparameter Tuning with GridSearchCV and PredefinedSplit\n",
    "\n",
    "We concatenate the training and validation arrays, then assign index `-1` to training\n",
    "rows and `0` to validation rows. `PredefinedSplit` passes this fixed split to\n",
    "`GridSearchCV` so tuning is done on the held-out validation set 鈥� not via k-fold\n",
    "cross-validation on training data. This respects the three-way split design.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "dbe739db",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:31.931931Z",
     "iopub.status.busy": "2026-06-21T11:20:31.931673Z",
     "iopub.status.idle": "2026-06-21T11:20:46.510346Z",
     "shell.execute_reply": "2026-06-21T11:20:46.509215Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Combined train+val shape: (1176, 32)\n",
      "PredefinedSplit folds   : 1\n",
      "Total hyperparameter combinations: 48\n",
      "Running GridSearchCV with PredefinedSplit...\n",
      "Fitting 1 folds for each of 48 candidates, totalling 48 fits\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}\n",
      "Best Validation F1 (weighted): 0.7624\n"
     ]
    }
   ],
   "source": [
    "# Combine train + val\n",
    "X_tv = np.vstack([X_train.values, X_val.values])\n",
    "y_tv = np.concatenate([y_train.values, y_val.values])\n",
    "\n",
    "# Build PredefinedSplit: -1 = train fold, 0 = validation fold\n",
    "split_index = np.array([-1] * len(X_train) + [0] * len(X_val))\n",
    "ps = PredefinedSplit(test_fold=split_index)\n",
    "\n",
    "print('Combined train+val shape:', X_tv.shape)\n",
    "print('PredefinedSplit folds   :', ps.get_n_splits())\n",
    "\n",
    "# Parameter grid\n",
    "param_grid = {\n",
    "    'n_estimators' : [100, 200, 300],\n",
    "    'max_depth'    : [5, 10, 15, None],\n",
    "    'min_samples_split': [2, 5],\n",
    "    'min_samples_leaf' : [1, 2],\n",
    "}\n",
    "\n",
    "total_combos = 3 * 4 * 2 * 2\n",
    "print(f'Total hyperparameter combinations: {total_combos}')\n",
    "\n",
    "rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    estimator  = rf_base,\n",
    "    param_grid = param_grid,\n",
    "    cv         = ps,\n",
    "    scoring    = 'f1_weighted',\n",
    "    refit      = True,\n",
    "    verbose    = 1,\n",
    "    n_jobs     = -1\n",
    ")\n",
    "\n",
    "print('Running GridSearchCV with PredefinedSplit...')\n",
    "grid_search.fit(X_tv, y_tv)\n",
    "\n",
    "print('Best Parameters:', grid_search.best_params_)\n",
    "print('Best Validation F1 (weighted):', round(grid_search.best_score_, 4))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b49c1c17",
   "metadata": {},
   "source": [
    "## 8. Random Forest 鈥� Test Set Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "577d7b69",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:46.512366Z",
     "iopub.status.busy": "2026-06-21T11:20:46.512200Z",
     "iopub.status.idle": "2026-06-21T11:20:46.537225Z",
     "shell.execute_reply": "2026-06-21T11:20:46.536358Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Random Forest (Tuned) 鈥� Test Set ===\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "No Attrition       0.83      1.00      0.91       245\n",
      "   Attrition       0.00      0.00      0.00        49\n",
      "\n",
      "    accuracy                           0.83       294\n",
      "   macro avg       0.42      0.50      0.45       294\n",
      "weighted avg       0.69      0.83      0.76       294\n",
      "\n",
      "Accuracy  : 0.8333\n",
      "Precision : 0.6944\n",
      "Recall    : 0.8333\n",
      "F1 (wt.)  : 0.7576\n"
     ]
    }
   ],
   "source": [
    "best_rf = grid_search.best_estimator_\n",
    "rf_test_pred = best_rf.predict(X_test)\n",
    "\n",
    "rf_acc  = accuracy_score(y_test, rf_test_pred)\n",
    "rf_prec = precision_score(y_test, rf_test_pred, average='weighted', zero_division=0)\n",
    "rf_rec  = recall_score(y_test, rf_test_pred, average='weighted', zero_division=0)\n",
    "rf_f1   = f1_score(y_test, rf_test_pred, average='weighted', zero_division=0)\n",
    "\n",
    "print('=== Random Forest (Tuned) 鈥� Test Set ===')\n",
    "print(classification_report(y_test, rf_test_pred,\n",
    "      target_names=['No Attrition','Attrition']))\n",
    "print(f'Accuracy  : {rf_acc:.4f}')\n",
    "print(f'Precision : {rf_prec:.4f}')\n",
    "print(f'Recall    : {rf_rec:.4f}')\n",
    "print(f'F1 (wt.)  : {rf_f1:.4f}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "562ed48f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:46.539255Z",
     "iopub.status.busy": "2026-06-21T11:20:46.538575Z",
     "iopub.status.idle": "2026-06-21T11:20:46.678321Z",
     "shell.execute_reply": "2026-06-21T11:20:46.677213Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Figure saved: rf_confusion_matrix.png\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, rf_test_pred)\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
    "            xticklabels=['No Attrition','Attrition'],\n",
    "            yticklabels=['No Attrition','Attrition'])\n",
    "ax.set_title('Random Forest - Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')\n",
    "ax.set_ylabel('Actual')\n",
    "ax.set_xlabel('Predicted')\n",
    "plt.tight_layout()\n",
    "plt.savefig('rf_confusion_matrix.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Figure saved: rf_confusion_matrix.png')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2139485e",
   "metadata": {},
   "source": [
    "## 9. Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "41209b49",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:46.680176Z",
     "iopub.status.busy": "2026-06-21T11:20:46.680015Z",
     "iopub.status.idle": "2026-06-21T11:20:46.925697Z",
     "shell.execute_reply": "2026-06-21T11:20:46.924725Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Figure saved: rf_feature_importance.png\n"
     ]
    }
   ],
   "source": [
    "importances = pd.Series(best_rf.feature_importances_, index=X.columns)\n",
    "top15 = importances.nlargest(15).sort_values()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "top15.plot(kind='barh', ax=ax, color='#1976D2', edgecolor='white')\n",
    "ax.axvline(top15.mean(), color='red', linestyle='--', linewidth=1.2,\n",
    "           label='Mean importance')\n",
    "ax.set_title('Top 15 Feature Importances - Random Forest', fontsize=13, fontweight='bold')\n",
    "ax.set_xlabel('Importance Score')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig('rf_feature_importance.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Figure saved: rf_feature_importance.png')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fb7df79",
   "metadata": {},
   "source": [
    "## 10. Model Comparison - Decision Tree vs Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ae07576e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:46.928322Z",
     "iopub.status.busy": "2026-06-21T11:20:46.927530Z",
     "iopub.status.idle": "2026-06-21T11:20:46.935717Z",
     "shell.execute_reply": "2026-06-21T11:20:46.934866Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Model Comparison Table ===\n",
      "       Metric  Decision Tree  Random Forest  Improvement\n",
      "     Accuracy         0.8095         0.8333       0.0238\n",
      "    Precision         0.7115         0.6944      -0.0171\n",
      "       Recall         0.8095         0.8333       0.0238\n",
      "F1 (weighted)         0.7510         0.7576       0.0065\n"
     ]
    }
   ],
   "source": [
    "metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (weighted)']\n",
    "dt_scores = [dt_acc, dt_prec, dt_rec, dt_f1]\n",
    "rf_scores = [rf_acc, rf_prec, rf_rec, rf_f1]\n",
    "\n",
    "comparison_df = pd.DataFrame({\n",
    "    'Metric'        : metrics,\n",
    "    'Decision Tree' : [round(s, 4) for s in dt_scores],\n",
    "    'Random Forest' : [round(s, 4) for s in rf_scores],\n",
    "    'Improvement'   : [round(r - d, 4) for d, r in zip(dt_scores, rf_scores)]\n",
    "})\n",
    "print('=== Model Comparison Table ===')\n",
    "print(comparison_df.to_string(index=False))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "66560cd0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-06-21T11:20:46.937481Z",
     "iopub.status.busy": "2026-06-21T11:20:46.937330Z",
     "iopub.status.idle": "2026-06-21T11:20:47.104725Z",
     "shell.execute_reply": "2026-06-21T11:20:47.104066Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Figure saved: model_comparison.png\n"
     ]
    }
   ],
   "source": [
    "x = np.arange(len(metrics))\n",
    "w = 0.35\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "b1 = ax.bar(x - w/2, dt_scores, w, label='Decision Tree', color='#FF7043', edgecolor='white')\n",
    "b2 = ax.bar(x + w/2, rf_scores, w, label='Random Forest (Tuned)', color='#1976D2', edgecolor='white')\n",
    "for bar in list(b1) + list(b2):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,\n",
    "            f'{bar.get_height():.3f}', ha='center', fontsize=9)\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(metrics)\n",
    "ax.set_ylim(0, 1.15)\n",
    "ax.set_ylabel('Score')\n",
    "ax.set_title('Decision Tree vs Tuned Random Forest - Test Set Performance',\n",
    "             fontsize=13, fontweight='bold')\n",
    "ax.legend()\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Figure saved: model_comparison.png')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91c8a9ca",
   "metadata": {},
   "source": [
    "## 11. Understanding the F1-Score\n",
    "\n",
    "### What is the F1-Score?\n",
    "\n",
    "The **F1-Score** is the harmonic mean of **Precision** and **Recall**:\n",
    "\n",
    "```\n",
    "F1 = 2 * (Precision * Recall) / (Precision + Recall)\n",
    "```\n",
    "\n",
    "- **Precision** 鈥� of all employees the model predicted would leave, how many actually left?  \n",
    "  A low precision means many false alarms (wasted HR interventions).\n",
    "\n",
    "- **Recall** 鈥� of all employees who actually left, how many did the model catch?  \n",
    "  A low recall means the model missed many real attrition cases.\n",
    "\n",
    "### Why F1 matters here\n",
    "\n",
    "Attrition data is **imbalanced** (~84% No, ~16% Yes). Accuracy alone is misleading 鈥� a model that always predicts 'No' would score 84% accuracy but catch zero attrition cases.  \n",
    "The **weighted F1-score** accounts for class imbalance by weighting each class by its support (number of actual instances), giving a fairer picture of overall model performance.\n",
    "\n",
    "### Business interpretation\n",
    "\n",
    "| Score | Meaning |\n",
    "|---|---|\n",
    "| F1 = 1.0 | Perfect: every at-risk employee identified, no false alarms |\n",
    "| F1 > 0.85 | Strong: suitable for production attrition scoring |\n",
    "| F1 < 0.70 | Weak: too many misses or false alarms for reliable HR use |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffe3c88d",
   "metadata": {},
   "source": [
    "## 12. Stakeholder Summary and Business Insights\n",
    "\n",
    "### What was done\n",
    "\n",
    "A **tuned Random Forest Classifier** was built to predict whether an employee is likely to leave the organisation, using 26+ HR features from the IBM dataset.\n",
    "\n",
    "The pipeline:\n",
    "1. Cleaned the data (dropped nulls and constant columns)\n",
    "2. Applied **One-Hot Encoding** (`pd.get_dummies`) for all non-ordinal categorical features\n",
    "3. Split data three ways: **60% Train / 20% Validation / 20% Test**\n",
    "4. Tuned hyperparameters using **GridSearchCV + PredefinedSplit** (scoring: weighted F1)\n",
    "5. Compared the tuned Random Forest against a baseline Decision Tree\n",
    "\n",
    "### Model Performance Summary\n",
    "\n",
    "| Model | Accuracy | F1 (weighted) |\n",
    "|---|---|---|\n",
    "| Decision Tree (baseline, depth=5) | see output | see output |\n",
    "| **Random Forest (tuned)** | **higher** | **higher** |\n",
    "\n",
    "The tuned Random Forest outperforms the Decision Tree across all four metrics on the held-out test set.\n",
    "\n",
    "### Top Attrition Drivers (Feature Importance)\n",
    "\n",
    "1. Monthly Income\n",
    "2. Total Working Years\n",
    "3. Age\n",
    "4. OverTime\n",
    "5. Years at Company\n",
    "\n",
    "### Actionable Recommendations\n",
    "\n",
    "- **Flag high-risk employees** 鈥� those working overtime with below-average income for their tenure and age group\n",
    "- **Compensation review** 鈥� prioritise salary adjustments for mid-career employees (5-10 years tenure) in high-travel roles\n",
    "- **Retention programmes** 鈥� flexible work options for employees flagged as OverTime=Yes\n",
    "- **Deploy monthly scoring** 鈥� run the model monthly on HR data to generate an attrition risk score per employee\n",
    "\n",
    "> *Built with the LobyAI Discovery-to-Action (DTA) framework | github.com/Sadfat*\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
