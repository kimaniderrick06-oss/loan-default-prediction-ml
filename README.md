# Loan Default Prediction Using Machine Learning

This project develops a machine learning classification system for predicting whether a customer is likely to default on a loan. The project compares Logistic Regression and Random Forest models and evaluates their ability to identify customers at risk of loan default.

## Project Overview

The project follows an end-to-end machine learning workflow:

- Data inspection and cleaning
- Exploratory Data Analysis (EDA)
- Feature and target separation
- Data preprocessing
- Train-test splitting
- Logistic Regression modeling
- Random Forest modeling
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Feature importance analysis

## Business Problem

Loan defaults can result in significant financial losses for lending institutions.

The business question addressed by this project is:

> How can machine learning be used to predict whether a customer is likely to default on a loan based on their financial, employment, and demographic characteristics?

## Project Objective

The objective is to develop and evaluate machine learning classification models that can predict loan default.

The project specifically aims to:

- Identify factors associated with loan default.
- Build Logistic Regression and Random Forest models.
- Compare the performance of both models.
- Use cross-validation to assess model consistency.
- Tune the Random Forest hyperparameters.
- Evaluate the final model using classification metrics.
- Identify the most important features used by the Random Forest model.

## Dataset

The dataset contains **1,212 customer records and 16 columns**.

The target variable is:

- `loan_default` — indicates whether a customer defaulted on their loan.

### Features

| Feature | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `monthly_income_ksh` | Monthly income in Kenyan Shillings |
| `employment_years` | Number of years employed |
| `credit_score` | Customer credit score |
| `loan_amount_ksh` | Loan amount in Kenyan Shillings |
| `loan_term_months` | Loan repayment period |
| `existing_loans` | Number of existing loans |
| `late_payments_last_12m` | Number of late payments in the previous 12 months |
| `debt_to_income_ratio` | Debt-to-income ratio |
| `savings_ksh` | Customer savings in Kenyan Shillings |
| `dependents` | Number of dependents |
| `employment_type` | Customer employment category |
| `education_level` | Customer education level |
| `housing_status` | Customer housing status |
| `loan_default` | Target variable indicating loan default |

## Data Cleaning & Preprocessing

The dataset was inspected for missing values, data types, and potential data-quality issues.

The preprocessing steps included:

- Separating features from the target variable.
- Removing `customer_id` because it is an identifier rather than a predictive feature.
- Splitting the dataset into training and testing sets.
- Handling missing numerical values using median imputation.
- Handling missing categorical values using the most frequent value.
- Standardizing numerical variables using `StandardScaler`.
- Encoding categorical variables using `OneHotEncoder`.

An **80/20 train-test split** was used with stratification to preserve the distribution of the target variable.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the dataset and investigate potential relationships between customer characteristics and loan default.

The analysis focused on variables including:

- Credit score
- Late payments
- Loan amount
- Savings
- Debt-to-income ratio
- Monthly income
- Employment characteristics

## Machine Learning Models

## Logistic Regression

Logistic Regression was used as the baseline classification model.

It was selected because it is a relatively simple and interpretable algorithm for binary classification.

## Random Forest

Random Forest was used as the second classification model.

It was selected because it can capture nonlinear relationships and interactions between different features.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Recall and F1 Score were given particular attention because correctly identifying customers who default is important in a loan-risk scenario.

## Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 67.80% | 72.22% | 79.05% | 75.48% |
| Random Forest | 70.76% | 74.53% | 81.08% | 77.67% |

The initial Random Forest model performed better than Logistic Regression across the four reported test-set metrics.

## Cross-Validation

Five-fold cross-validation was used to evaluate the consistency of model performance.

F1 Score was used as the scoring metric.

| Model | Mean Cross-Validation F1 |
|---|---:|
| Logistic Regression | 79.62% |
| Random Forest | 78.29% |

Although Random Forest performed better on the initial test set, Logistic Regression achieved a slightly higher mean cross-validation F1 score.

## Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model.

The best parameters were:

```text
n_estimators = 200
max_depth = 5
min_samples_split = 10
## Final Model Results

After hyperparameter tuning, the Random Forest model was evaluated on the unseen test dataset.

| Metric | Final Random Forest |
|---|---:|
| Accuracy | 66.95% |
| Precision | 69.44% |
| Recall | 84.46% |
| F1 Score | 76.22% |
| ROC-AUC | 72.49% |

The final Random Forest model achieved a **recall of 84.46%**, meaning it correctly identified a large proportion of customers who actually defaulted on their loans.

The model achieved an **ROC-AUC score of 72.49%**, indicating a reasonable ability to distinguish between customers who defaulted and those who did not.


## Feature Importance

The Random Forest model was used to identify the features that contributed most to its predictions.

| Feature | Importance |
|---|---:|
| Credit Score | 32.97% |
| Late Payments – Last 12 Months | 17.85% |
| Loan Amount | 7.12% |
| Savings | 6.89% |
| Debt-to-Income Ratio | 6.56% |
| Age | 6.17% |
| Monthly Income | 5.64% |
| Employment Years | 4.77% |

**Credit score** was the most important feature, followed by **late payments during the previous 12 months**.

> Feature importance shows how useful a variable was for the model's predictions. It does not necessarily mean that the feature causes loan default.


## Key Findings

- Random Forest initially outperformed Logistic Regression on the test-set accuracy, precision, recall, and F1 score.
- Logistic Regression achieved a slightly higher mean cross-validation F1 score than the initial Random Forest.
- Hyperparameter tuning improved the Random Forest's cross-validation performance.
- The best Random Forest hyperparameters were `n_estimators=200`, `max_depth=5`, and `min_samples_split=10`.
- Credit score was the most important predictive feature in the Random Forest model.
- Late payments during the previous 12 months were the second most important feature.
- The final Random Forest achieved **84.46% recall**, making it relatively effective at identifying customers who actually defaulted.
- The final model achieved an **F1 score of 76.22%** and an **ROC-AUC of 72.49%**.
- Accuracy alone was not used to determine model performance because correctly identifying customers who default is particularly important in a loan-risk context.


## Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization
- **Scikit-learn** – Data preprocessing, model training, cross-validation, hyperparameter tuning, and evaluation
- **Jupyter Notebook** – Development and analysis environment
- **Git** – Version control
- **GitHub** – Project repository and documentation
