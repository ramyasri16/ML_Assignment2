Credit Approval Prediction Dashboard

a. Problem Statement
The objective of this project is to automate and streamline the credit card approval process. In the financial sector, evaluating credit applications manually is a resource-intensive task. By using supervised machine learning algorithms, this application predicts the outcome of a credit request (Approved + or Rejected -) based on historical applicant data. This enables faster, data-driven decision-making while minimizing human bias and error.

b. Dataset Description
The application uses the UCI Credit Approval Dataset (also known as the crx dataset). As seen in the app preview, the data consists of 15 anonymized features (A1–A15) to protect the confidentiality of the applicants.

Total Records: 690 applications.

Target Variable: + (Approved) or - (Rejected).

Features:

Categorical Features: A1 (Gender/Status), A4, A5, A6 (Occupation), A7 (Ethnicity), A9 (Prior Default), A10 (Employed), A12, A13.

Numerical Features: A2 (Age), A3 (Debt), A8 (Years Employed), A11 (Credit Score), A14 (Zip Code), A15 (Income).

Data Characteristics: The dataset contains a mix of categorical and continuous values, with some missing values (represented as ? in raw format) which are handled via imputation during the preprocessing stage.

c. Models Used
Six machine learning models were trained and integrated into the dashboard. Below is the comparison of their performance on the testing set:

### Comparison Table: Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.833 | 0.894 | 0.836 | 0.824 | 0.830 | 0.667 |
| **Decision Tree** | 0.768 | 0.768 | 0.757 | 0.779 | 0.768 | 0.537 |
| **kNN** | 0.855 | 0.883 | 0.808 | 0.926 | 0.863 | 0.718 |
| **Naive Bayes** | 0.754 | 0.830 | 0.698 | 0.882 | 0.779 | 0.527 |
| **Random Forest** | 0.862 | 0.917 | 0.836 | 0.897 | 0.865 | 0.727 |
| **XGBoost** | **0.961** | **0.975** | **0.961** | **0.961** | **0.961** | **0.921** |


### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed as a stable baseline with high AUC (0.894), indicating excellent class separation for a linear model. |
| **Decision Tree** | Showed the lowest overall performance; likely due to its simplicity compared to the ensemble methods. |
| **kNN** | Demonstrated excellent Recall (0.926); it is highly effective at identifying positive credit approval cases. |
| **Naive Bayes** | Provided competitive recall but suffered from lower precision (0.698), leading to more false approval predictions. |
| **Random Forest** | A very strong performer with a high F1-score (0.865), showing the power of bagging in reducing variance. |
| **XGBoost** | **Overall Best Performer.** Achieved a superior Accuracy of 96.1% and an MCC of 0.921, significantly outperforming all other models through gradient boosting. |
