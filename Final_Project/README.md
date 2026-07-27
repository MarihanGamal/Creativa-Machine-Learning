# Telco Customer Churn Prediction

A complete, end-to-end machine learning graduation project that predicts whether a
telecom customer will churn, built entirely on the real **IBM Telco Customer Churn**
dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) — no synthetic or generated data is
used anywhere in this project.

## Project Overview

Customer churn — when a customer stops using a company's service — is one of the most
important metrics for subscription-based businesses. This project builds a full
classification pipeline to predict churn from customer account and service attributes,
and ships that model behind an interactive Streamlit web app.

## Dataset

- **Source file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7,043 customers
- **Columns:** 21 (20 features + target)
- **Target variable:** `Churn` (`Yes` / `No`)
- **Class balance:** ~73.5% No / ~26.5% Yes (moderately imbalanced)

Feature groups:
| Type | Examples |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Identifier | `customerID` (dropped before modeling) |

## Project Structure

```
.
├── Telco_Customer_Churn_Prediction.ipynb   # Full, executed analysis & modeling notebook
├── app.py                                  # Streamlit web app for live predictions
├── model_utils.py                          # Shared lightweight ColumnSelector transformer
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── preprocessor.pkl                        # Fitted ColumnTransformer (scaling + encoding)
├── saved_model.pkl                         # Final tuned model pipeline
└── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Original dataset
```

## Methodology

The notebook (`Telco_Customer_Churn_Prediction.ipynb`) walks through every stage of the
pipeline, from raw data to a deployable model:

1. **Data Exploration** — shape, dtypes, summary statistics, target distribution.
2. **Missing Values** — `TotalCharges` was stored as text; converting it to numeric
   revealed 11 blank values, all belonging to customers with `tenure == 0` (brand-new,
   not yet billed). These were filled with `0` rather than dropped.
3. **Duplicate Handling** — no fully duplicated rows exist. 22 rows share identical
   attributes when `customerID` is excluded, but since every `customerID` is unique they
   represent distinct real customers, not duplicate records, and were kept.
4. **Data Cleaning** — dropped the non-predictive `customerID` identifier.
5. **Exploratory Data Analysis** — churn rate broken down by contract type, internet
   service, and payment method; distribution and correlation plots for numeric features.
6. **Outlier Analysis** — IQR method applied to `tenure`, `MonthlyCharges`, `TotalCharges`;
   no outliers detected (all values fall within naturally bounded business ranges).
7. **Encoding & Scaling** — a `ColumnTransformer` applies median imputation +
   `StandardScaler` to numeric features, and most-frequent imputation + `OneHotEncoder`
   to categorical features. Fit only on the training split to prevent data leakage.
8. **Train/Test Split** — 80/20, stratified on `Churn`.
9. **Feature Selection** — `SelectFromModel` with a Random Forest ranks feature
   importance; features above the median importance are kept (46 → 23 features).
10. **Cross-Validation & Model Comparison** — 5-fold stratified CV (scored on F1) across:
    Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
    AdaBoost, and (if installed) XGBoost, LightGBM, CatBoost.
11. **Hyperparameter Tuning** — `GridSearchCV` (5-fold, F1-scored) on the top 3 models
    from the comparison step.
12. **Final Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
    ROC curve, precision-recall curve, and classification report on the held-out test set.
13. **Feature Importance** — ranked plot of the most influential predictors.
14. **Model Persistence** — `preprocessor.pkl` and `saved_model.pkl` saved with `joblib`.

Class imbalance is handled via `class_weight='balanced'` (and `scale_pos_weight` for
XGBoost) rather than synthetic oversampling (e.g. SMOTE), to keep the project strictly
grounded in the real, provided dataset.

## Results

The best-performing model selected after cross-validation and hyperparameter tuning was
**Logistic Regression**, evaluated on the untouched 20% test set:

| Metric | Score |
|---|---|
| Accuracy | 0.7402 |
| Precision | 0.5070 |
| Recall | 0.7781 |
| F1 Score | 0.6139 |
| ROC-AUC | 0.8337 |

The top churn drivers identified were **tenure**, **contract type**, **monthly charges**,
and the **absence of tech support / online security** add-ons — consistent with common
telecom churn behavior (new, month-to-month, higher-paying customers without support
add-ons churn the most).

> Recall was prioritized over precision when selecting hyperparameters, since in a churn
> use case, failing to flag a customer who *will* churn (a false negative) is typically
> more costly to the business than a false alarm.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore the notebook
Open `Telco_Customer_Churn_Prediction.ipynb` in Jupyter to review the full analysis,
or re-run all cells end-to-end to regenerate `preprocessor.pkl` and `saved_model.pkl`
from scratch:
```bash
jupyter notebook Telco_Customer_Churn_Prediction.ipynb
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
This opens a browser form where you can enter a customer's attributes and get a live
churn prediction with probability.

## Notes

- `model_utils.py` defines a lightweight `ColumnSelector` transformer used inside
  `saved_model.pkl`. It must remain in the same directory as `app.py` (and be importable
  from the notebook) since it is required to unpickle the saved model.
- XGBoost, LightGBM, and CatBoost are optional. The notebook detects their availability
  automatically at runtime and includes them in model comparison only if installed —
  the pipeline runs correctly with or without them.

## Author's Note

This project was built for academic/graduation submission purposes using only the
provided real-world dataset, with no synthetic or generated records introduced at any
stage of data loading, cleaning, exploration, or modeling.
