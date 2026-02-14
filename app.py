import streamlit as st
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Page Config
st.set_page_config(page_title="Credit Approval Prediction", layout="wide")

st.title("Credit Approval Prediction App")
st.markdown("Predicts whether a credit card application will be Approved (+) or Rejected (-).")

# Sidebar
st.sidebar.header("Configuration")

# 1. Load Metrics
st.subheader("Model Performance Comparison")
try:
    metrics_df = pd.read_csv('models/metrics.csv')
    st.dataframe(metrics_df.style.highlight_max(axis=0))
except:
    st.warning("Please run 'train_models.py' first.")

# 2. Model Selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# Load Model and Preprocessing Artifacts
try:
    with open(f'models/{model_name.replace(" ", "_")}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/imputer_num.pkl', 'rb') as f:
        imputer_num = pickle.load(f)
    with open('models/imputer_cat.pkl', 'rb') as f:
        imputer_cat = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (No Header, 16 Columns)", type=["csv"])

if uploaded_file is not None:
    # Read without header, matching the training format
    col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
    input_df = pd.read_csv(uploaded_file, header=None, names=col_names, na_values='?')
    
    st.write(f"### Uploaded Data ({input_df.shape[0]} rows)")
    st.dataframe(input_df.head())

    if st.button("Predict"):
        try:
            # Separate Features and Target
            X_new = input_df.drop('class', axis=1)
            y_true_raw = input_df['class']
            
            # --- Preprocessing Pipeline (Must match training) ---
            categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
            numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
            
            # 1. Impute
            X_new[numerical_cols] = imputer_num.transform(X_new[numerical_cols])
            X_new[categorical_cols] = imputer_cat.transform(X_new[categorical_cols])
            
            # 2. Encode Categoricals
            # Note: Unseen labels in test data that weren't in training can cause errors with basic LabelEncoder.
            # We handle this by assigning a default (e.g., 0) or handling exception, but for assignment dataset usually it matches.
            for col in categorical_cols:
                le = encoders[col]
                # Helper to handle unseen labels
                X_new[col] = X_new[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
                le_classes = le.classes_.tolist()
                # If unknown, map to most frequent (first class usually) or special handling. 
                # Simplest for this assignment: Map to valid class 0 if unknown
                X_new[col] = X_new[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

            # 3. Scale (if required)
            if model_name in ["Logistic Regression", "KNN"]:
                X_processed = scaler.transform(X_new)
            else:
                X_processed = X_new

            # 4. Predict
            y_pred = model.predict(X_processed)
            
            # Encode true labels for metrics
            y_true = target_encoder.transform(y_true_raw)

            # Display Results
            st.subheader(f"Results for {model_name}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.write("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Check: Does your CSV have 16 columns? Are missing values marked as '?'")

else:
    st.info("Please upload a CSV file matching the 'crx.data' format (16 columns, no header).")
