import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Credit Predictor", layout="wide")

st.title("Credit Approval Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("App Controls")

if os.path.exists('crx.csv'):
    with open("crx.csv", "rb") as f:
        st.sidebar.download_button("Download Sample CSV", data=f, file_name="sample_data.csv")

f = st.sidebar.file_uploader("Choose a CSV file", type="csv")
models = ["Random Forest", "Logistic Regression", "XGBoost", "KNN", "Decision Tree", "Naive Bayes"]
selected_model = st.sidebar.selectbox("Select ML Model", models)

if f:
    col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'target']
    df = pd.read_csv(f, header=None, names=col_names, na_values="?")
    
    # Fill missing values so encoding doesn't crash
    df = df.ffill().bfill()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            # 1. Load Model
            fname = selected_model.lower().replace(' ', '_') + ".pkl"
            path = os.path.join("models", fname)
            if not os.path.exists(path):
                path = os.path.join("models", selected_model.replace(' ', '_') + ".pkl")
            
            model = joblib.load(path)
            
            # 2. Preprocessing: Convert Strings to Numbers
            X = df.drop(columns=['target']).copy()
            y = df['target']

            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Ensure y is also encoded if it contains '+' or '-'
            if y.dtype == 'object':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))

            # 3. Predict
            preds = model.predict(X)
            
            # 4. Results
            st.success(f"Results for {selected_model}")
            left, right = st.columns(2)
            
            with left:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y, preds)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                
            with right:
                st.write("**Metrics Report**")
                metrics = classification_report(y, preds, output_dict=True)
                st.table(pd.DataFrame(metrics).transpose())

        except Exception as err:
            st.error(f"Error: {err}")
            st.info("This error usually happens when the data contains text that the model can't process directly.")
else:
    st.write("Please upload the `crx.csv` file to begin.")
