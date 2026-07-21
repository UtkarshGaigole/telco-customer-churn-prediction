# Dataset: Telco Customer Churn

## Source
- **Link:** <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- **Repository:** Kaggle
- **Original provider:** IBM Sample Data Sets

## Description
This dataset contains customer records from a fictional telecommunications company, provided
by IBM as a sample dataset. Each row represents one customer and includes demographic
information, subscribed services, account and billing details, and whether the customer left
the company within the last month (churn). In this project, the dataset is used to train and
evaluate classification models that predict customer churn, helping identify which factors
most strongly influence a customer's decision to leave.

## Dataset Details
| Property | Value |
|---|---|
| **File** | `Telco-Customer-Churn.csv` |
| **Rows (raw)** | 7,043 customers |
| **Columns (raw)** | 21 |
| **Rows (after cleaning)** | 7,032 (11 rows with blank `TotalCharges` removed — 0.16%) |
| **Target variable** | `Churn` (Yes / No → encoded 1 / 0) |
| **Class balance** | 73.4% No / 26.6% Yes (imbalanced) |
| **Problem type** | Binary classification |
| **Missing values** | 11 blank strings in `TotalCharges` (hidden — not detected by `isnull()`; found via `pd.to_numeric(errors="coerce")`) |

## Feature Overview
- **Identifier:** customerID — *removed during preprocessing (no predictive value)*
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account information:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies

## Engineered Features (added in `src/preprocessing.py`)
| Feature | Definition | Rationale |
|---|---|---|
| `TenureGroup` | tenure binned into 0-1 / 1-2 / 2-4 / 4+ years | Captures non-linear lifetime-stage effects; first-year customers churn most |
| `AvgChargesPerMonth` | `TotalCharges / tenure` | Average spend per month — ranks among the top predictors of the final model |

## Why This Dataset?
We chose this dataset because customer churn is a realistic and widely relevant business
problem in the telecom industry, where retaining existing customers is more cost-effective
than acquiring new ones. The dataset is well-suited for a complete machine learning workflow:
it has a manageable size (7,043 rows, 21 features), a clearly defined binary target variable
(Churn), and a good mix of categorical and numerical features — which allows us to demonstrate
encoding, scaling, feature selection, and model comparison. It also contains realistic data
issues, such as hidden missing values in TotalCharges, giving us the opportunity to show
meaningful preprocessing decisions.

## How to Load
From the repository root:
```python
import pandas as pd
df = pd.read_csv("data/Telco-Customer-Churn.csv")
```
Or use the project's cleaning function (recommended — applies all documented preprocessing):
```python
from src.preprocessing import load_and_clean
df = load_and_clean()
```
