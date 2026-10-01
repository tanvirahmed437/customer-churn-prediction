# Customer Churn Prediction & Retention Analytics

## Project Overview

This project focuses on predicting customer churn using machine learning. The goal is to identify customers who are more likely to leave a telecommunications service and analyze the factors associated with customer churn.

The project uses the **IBM Telco Customer Churn dataset** and follows an end-to-end machine learning workflow, from data cleaning and exploratory analysis to model training, evaluation, hyperparameter tuning, and customer risk scoring.

## Objectives

* Analyze customer churn patterns
* Identify important factors associated with churn
* Build machine learning classification models
* Compare different classification algorithms
* Tune the final model using cross-validation
* Predict customer churn probability
* Categorize customers into different churn-risk levels

## Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains customer information including:

* Demographic information
* Account information
* Contract details
* Internet services
* Additional services
* Payment methods
* Monthly charges
* Total charges
* Customer tenure
* Churn status

The target variable is **Churn**, which indicates whether a customer left the service.

## Project Workflow

### 1. Data Cleaning

* Loaded the dataset using Pandas
* Inspected dataset structure and data types
* Checked for missing values and duplicates
* Converted `TotalCharges` into a numeric format
* Handled missing values
* Removed the `customerID` identifier

### 2. Exploratory Data Analysis

Analyzed relationships between churn and customer characteristics such as:

* Contract type
* Tenure
* Monthly charges
* Internet service
* Payment method
* Tech support
* Senior citizen status
* Gender
* Customer services

Correlation analysis was also performed for numerical variables.

### 3. Feature Engineering

Created additional features including:

* `num_services` — number of additional services used by a customer
* `is_long_term` — indicator for customers with 24 or more months of tenure

### 4. Data Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were encoded using `OneHotEncoder`.

A Scikit-learn `ColumnTransformer` and pipeline were used to combine preprocessing and model training.

### 5. Machine Learning Models

The following classification models were evaluated:

* Dummy Classifier
* Logistic Regression
* Decision Tree
* Random Forest

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### 6. Model Validation and Tuning

Cross-validation was used to evaluate model performance.

The Random Forest model was further optimized using `GridSearchCV` to search for suitable hyperparameters.

### 7. Final Model

The tuned Random Forest model was used as the final churn prediction model.

The final model was evaluated on a separate test set.

### 8. Feature Importance

Random Forest feature importance was analyzed to identify features that contributed most to the model's predictions.

### 9. Customer Risk Scoring

The final model generates a churn probability for each customer.

Customers were grouped into:

* **Low Risk**
* **Medium Risk**
* **High Risk**

This provides a simple way to prioritize customers for further analysis or potential retention activities.

## Results

The final tuned Random Forest model achieved the following performance on the test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7906 |
| Precision | 0.6639 |
| Recall    | 0.4278 |
| F1 Score  | 0.5203 |
| ROC-AUC   | 0.8396 |

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Dataset
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── figures/
│   ├── feature_importance.png
│   └── risk_distribution.png
│
├── models/
│   └── churn_model.pkl
│
└── README.md
```

## Business Application

A churn prediction model can help a telecommunications company identify customers with higher predicted churn probability.

The predictions can be used to prioritize customers for further investigation and potential retention strategies.

The model identifies statistical patterns in historical data and does not establish that a particular customer characteristic causes churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## Future Improvements

Potential improvements include:

* Optimizing the probability threshold based on business costs
* Probability calibration
* Adding additional customer behavior features
* Cost-sensitive learning
* Testing additional machine learning algorithms
* Evaluating the model on new customer data

## Author

**TANVIR AHMED**

This project was developed as a machine learning project to practice the end-to-end data science workflow, including data analysis, feature engineering, model development, evaluation, and model preparation.
