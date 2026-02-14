import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Basic page config
st.set_page_config(page_title="Credit Predictor", layout="wide")

st.title("Credit Approval Prediction Dashboard")
st.markdown("---")

# Sidebar setup
st.sidebar.title("App Controls")

# Simple download for mentors
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
    # Load and show data
    df = pd.read_csv(f, header=None, na_values="?")
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            # Load the saved model file
            fname = selected_model.lower().replace(' ', '_') + ".pkl"
            path = os.path.join("models", fname)
            
            if not os.path.exists(path):
                st.error(f"Couldn't find {fname} in models folder.")
            else:
                model = joblib.load(path)
                
                # Split features and label
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                preds = model.predict(X)
                
                st.success(f"Results for {selected_model}")
                
                # Layout for charts
                left, right = st.columns(2)
                
                with left:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y, preds)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                    
                with right:
                    st.write("**Metrics Report**")
                    metrics = classification_report(y, preds, output_dict=True)
                    st.table(pd.DataFrame(metrics).transpose())

        except Exception as err:
            st.error(f"Something went wrong: {err}")
else:
    st.write("Please upload the `crx.csv` file in the sidebar to run the analysis.")
