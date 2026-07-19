# Dataset: Telco Customer Churn

## Source
- **Link:** <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- **Repository:** Kaggle
- **Original provider:** IBM Sample Data Sets

## Description
This dataset contains customer records from a fictional telecommunications company, provided by IBM as a sample dataset. Each row represents one customer and includes demographic information, subscribed services, account and billing details, and whether the customer left the company within the last month (churn). In this project, the dataset is used to train and evaluate classification models that predict customer churn, helping identify which factors most strongly influence a customer's decision to leave.

## Dataset Details
| Property | Value |
|---|---|
| **File** | `Telco-Customer-Churn` |
| **Rows** | 7,043 customers |
| **Features** | 21 columns |
| **Target variable** | `Churn` (Yes / No) |
| **Problem type** | Binary classification |
| **Missing values** | <e.g., 11 blank values in `TotalCharges`> |

## Feature Overview
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account information:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

## Why This Dataset?
We chose this dataset because customer churn is a realistic and widely relevant business problem in the telecom industry, where retaining existing customers is more cost-effective than acquiring new ones. The dataset is well-suited for a complete machine learning workflow: it has a manageable size (7,043 rows, 21 features), a clearly defined binary target variable (Churn), and a good mix of categorical and numerical features — which allows us to demonstrate encoding, scaling, feature selection, and model comparison. It also contains realistic data issues, such as missing values in TotalCharges, giving us the opportunity to show meaningful preprocessing decisions.

## How to Load
```python
import pandas as pd
df = pd.read_csv('data/Telco-Customer-Churn.csv')
```
