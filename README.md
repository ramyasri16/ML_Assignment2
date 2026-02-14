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

Comparison Table: Evaluation Metrics

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.85	0.92	0.84	0.87	0.85	0.70
Decision Tree	0.82	0.81	0.81	0.82	0.81	0.63
kNN	0.84	0.88	0.83	0.85	0.84	0.68
Naive Bayes	0.79	0.89	0.74	0.88	0.80	0.60
Random Forest	0.88	0.94	0.87	0.90	0.88	0.76
XGBoost	0.89	0.95	0.88	0.91	0.89	0.78


Observations on Model Performance

ML Model Name	Observation about model performance
Logistic Regression	Achieved solid performance as a baseline model; its high AUC suggests a very good ability to distinguish between classes.
Decision Tree	Provided high interpretability but had the lowest AUC, suggesting it may be slightly overfitting to the training noise.
kNN	Showed improved accuracy after numerical scaling; it effectively captured local patterns in the applicant features.
Naive Bayes	Maintained high recall (0.88), making it effective at catching potential approvals, though it had lower overall precision.
Random Forest	Demonstrated high stability and accuracy by aggregating multiple trees, significantly reducing variance compared to a single Decision Tree.
XGBoost	Best Performer. It achieved the highest Accuracy (89%) and MCC (0.78), proving to be the most robust model for this specific dataset.
