import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Page configuration
st.set_page_config(page_title="Credit Predictor", layout="wide")

st.title("Credit Approval Prediction Dashboard")
st.markdown("---")

# Sidebar setup
st.sidebar.title("App Controls")

if os.path.exists('crx.csv'):
    with open("crx.csv", "rb") as f:
        st.sidebar.download_button(
            "Download Sample CSV",
            data=f,
            file_name="sample_data.csv"
        )
else:
    st.sidebar.info("Upload your own data to begin.")

# File uploader
f = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Model selection
models = ["Random Forest", "Logistic Regression", "XGBoost", "KNN", "Decision Tree", "Naive Bayes"]
selected_model = st.sidebar.selectbox("Select ML Model", models)

if f:
    
    col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'target']
    
    df = pd.read_csv(f, header=None, names=col_names, na_values="?")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            # 1. Handle Model Path (Checking both lowercase and Title_Case)
            fname = selected_model.lower().replace(' ', '_') + ".pkl"
            alt_fname = selected_model.replace(' ', '_') + ".pkl"
            
            path = os.path.join("models", fname)
            if not os.path.exists(path):
                path = os.path.join("models", alt_fname)

            if not os.path.exists(path):
                st.error(f"Model file not found in 'models' folder.")
            else:
                model = joblib.load(path)
                
                # 2. Split features and label
                X = df.drop(columns=['target'])
                y = df['target']
                
                # 3. Predict
                preds = model.predict(X)
                
                st.success(f"Results for {selected_model}")
                
                # 4. Visualization Layout
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
                    report_df = pd.DataFrame(metrics).transpose()
                    st.table(report_df)

        except Exception as err:
            st.error(f"Something went wrong: {err}")
            st.info("Ensure your model version matches the environment (Scikit-Learn 1.6.1)")
else:
    st.write("Please upload the `crx.csv` file in the sidebar to run the analysis.")
