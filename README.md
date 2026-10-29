# Telco Customer Churn Prediction

Predicting which telecom customers are about to leave, using the IBM Telco Customer Churn
dataset and a complete, leakage-safe scikit-learn workflow.

Practical exam project for **Statistics and Machine Learning**, M.Sc. Applied Data Science and
Artificial Intelligence, SRH University of Applied Sciences Heidelberg.

## Results

Four classifiers were compared, each wrapped in a preprocessing pipeline and tuned with
`GridSearchCV` (5-fold stratified cross-validation, optimising F1 on the churn class).
Scores are on a held-out, stratified 20% test set (1,407 customers).

| Model | Baseline F1 | Tuned F1 | Tuned Recall | Tuned Precision | Tuned Accuracy |
|---|---|---|---|---|---|
| **Random Forest** (final model) | 0.555 | **0.630** | 0.762 | 0.537 | 0.762 |
| Logistic Regression | 0.573 | 0.612 | 0.783 | 0.502 | 0.736 |
| Decision Tree | 0.478 | 0.607 | 0.781 | 0.497 | 0.731 |
| KNN | 0.556 | 0.598 | 0.602 | 0.595 | 0.785 |

- The tuned Random Forest reaches **F1 ≈ 0.63, churn recall ≈ 0.76 and ROC AUC ≈ 0.84**. It
  identifies about three out of four customers who are about to churn.
- Tuning selected `class_weight="balanced"` for three of the four models, which raised churn
  recall from about 0.5 to about 0.76–0.78 at the cost of lower precision.
- The strongest churn drivers are contract type (month-to-month), tenure, charges, payment by
  electronic check, fiber-optic internet, and having no online security or tech support.

Full numbers: [`results/model_comparison.csv`](results/model_comparison.csv). Plots:
[`results/plots/`](results/plots/).

## Workflow

1. **Data cleaning**: 11 hidden blank values in `TotalCharges` are found and removed (0.16%);
   `customerID` is dropped.
2. **Exploratory and statistical analysis**: target balance (73% / 27%), distributions,
   correlations and an outlier check.
3. **Feature engineering**: `TenureGroup` (tenure bins) and `AvgChargesPerMonth`.
4. **Feature assessment**: Mutual Information, computed on the training set only.
5. **Modelling**: Logistic Regression, Decision Tree, Random Forest and KNN, each inside a
   `Pipeline` with scaling and one-hot encoding, so preprocessing is fitted on training data only.
6. **Tuning and validation**: `GridSearchCV` with 5-fold stratified CV, a train-vs-test
   overfitting check, and final evaluation on the held-out test set.
7. **Interpretation**: confusion matrix, ROC curves, class-wise metrics and feature importance.

## Repository structure

```
├── data/
│   ├── Telco-Customer-Churn.csv   # raw dataset (7,043 rows, 21 columns)
│   └── README.md                  # dataset description and source
├── notebooks/
│   └── ML_project_final_exam_v3.ipynb   # main analysis notebook
├── src/
│   ├── preprocessing.py           # loading, cleaning, feature engineering, split, preprocessor
│   └── evaluation.py              # models, tuning, evaluation and plotting functions
├── results/
│   ├── model_comparison.csv       # baseline vs tuned metrics
│   └── plots/                     # saved figures
├── requirements.txt
├── environment.yml
└── random_seed.txt                # global seed (42)
```

## How to run

Tested with Python 3.12.

```bash
git clone https://github.com/UtkarshGaigole/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# pip
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate telco-churn

jupyter notebook notebooks/ML_project_final_exam_v3.ipynb
```

Run the notebook top to bottom (Kernel → Restart & Run All). Every model uses
`random_state=42`, so the results above are reproduced exactly.

The cleaning and modelling steps are also available as functions in `src/`:

```python
from src.preprocessing import load_and_clean, split_data, build_preprocessor

df = load_and_clean()  # run from the repository root
X_train, X_test, y_train, y_test = split_data(df)
```

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle, IBM Sample Data Sets): a fictional telecom company's customers, with demographics,
subscribed services, account and billing details, and whether they churned in the last month.
See [`data/README.md`](data/README.md) for details.

## Team

Group project by Jasmeet Kaur, Aditya Sanjay Pawar, Sreelakshmi Koyambrath Sivaraman,
Utkarsh Anil Gaigole and Kalaivani Sivakumar. Individual contributions are listed in
section 12 of the notebook.

Supervisors: Prof. Dr. Mehrdad Jalali, Dr. Ghadeer Mobasher.
