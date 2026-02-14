import pandas as pd
import numpy as np
import pickle
import os

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics & Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Setup Directory
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load Dataset
# The dataset has no headers, so we define them based on A1-A16 convention
col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
# Replace '?' with NaN immediately upon loading
data = pd.read_csv('crx.csv', header=None, names=col_names, na_values='?')

# 2. Preprocessing

# Separate Target
X = data.drop('class', axis=1)
y = data['class']

# Encode Target (+ = 1, - = 0)
le_target = LabelEncoder()
y = le_target.fit_transform(y) 
# Save target encoder to inverse transform later if needed
with open('models/target_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)

# Handling Missing Values & Encoding Categorical Features
# We need to save the encoders to apply the exact same transformation to the Uploaded CSV in the App
categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']

# Impute Missing Values
# Numeric -> Mean
imputer_num = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

# Categorical -> Most Frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

# Label Encoding for Categorical Columns
# We use a dictionary to store an encoder for each column
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save Imputers and Encoders dictionary
with open('models/imputer_num.pkl', 'wb') as f:
    pickle.dump(imputer_num, f)
with open('models/imputer_cat.pkl', 'wb') as f:
    pickle.dump(imputer_cat, f)
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Important for KNN/Logistic)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 4. Train and Evaluate
results = []
print(f"{'Model':<20} | {'Acc':<5} | {'AUC':<5} | {'Prec':<5} | {'Rec':<5} | {'F1':<5} | {'MCC':<5}")
print("-" * 75)

for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name, "Accuracy": acc, "AUC": auc, "Precision": prec, 
        "Recall": rec, "F1": f1, "MCC": mcc
    })
    
    print(f"{name:<20} | {acc:.3f} | {auc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {mcc:.3f}")

    # Save Model
    with open(f'models/{name.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save metrics to CSV for the App
results_df = pd.DataFrame(results)
results_df.to_csv('models/metrics.csv', index=False)
print("\nTraining complete. Models and artifacts saved to 'models/' folder.")
