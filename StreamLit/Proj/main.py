import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from models import train_model

# Set the style for Seaborn plots
sns.set_style('whitegrid')

# Title of the application
st.title("Machine Learning Model Performance Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Data Overview", "Model Training", "Visualizations"])

# Load the dataset
@st.cache
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if page == "Home":
    st.header("Welcome to the ML Dashboard")
    st.write("""
    This application allows you to upload a dataset, preprocess it, 
    train different machine learning models, and visualize the results.
    Use the sidebar to navigate through different sections.
    """)
    
elif page == "Data Overview":
    st.header("Data Overview")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state['data'] = df
        st.subheader("Here is the dataframe")
        st.dataframe(df.head())

        st.subheader("Data Shape")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        st.subheader("Descriptive Statistics")
        st.write(df.describe())

elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state['data'] is not None:
        df = st.session_state['data']

        # Preprocess the data
        def preprocess_data(df):
            # Encode categorical variables
            label_encoders = {}
            for column in df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            return df, label_encoders

        # Model Selection and Training
        model_name = st.selectbox("Select a model", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"])
        
        if st.button("Train Model"):
            df_preprocessed, label_encoders = preprocess_data(df)
            X = df_preprocessed.drop(columns=['Drug'])
            y = df_preprocessed['Drug']
            model, metrics = train_model(model_name, X, y)

            st.subheader("Model Performance:")
            # Display metrics in markdown format
            performance_metrics = f"""
            **Classification Report:**

            | Metric        | Class 0  | Class 1  | Class 2  | Class 3  | Class 4  | Macro Avg | Weighted Avg |
            |---------------|----------|----------|----------|----------|----------|-----------|--------------|
            | Precision     | {metrics['0']['precision']:.2f}    | {metrics['1']['precision']:.2f}    | {metrics['2']['precision']:.2f}    | {metrics['3']['precision']:.2f}    | {metrics['4']['precision']:.2f}    | {metrics['macro avg']['precision']:.2f}     | {metrics['weighted avg']['precision']:.2f}      |
            | Recall        | {metrics['0']['recall']:.2f}       | {metrics['1']['recall']:.2f}       | {metrics['2']['recall']:.2f}       | {metrics['3']['recall']:.2f}       | {metrics['4']['recall']:.2f}       | {metrics['macro avg']['recall']:.2f}      | {metrics['weighted avg']['recall']:.2f}       |
            | F1-Score      | {metrics['0']['f1-score']:.2f}     | {metrics['1']['f1-score']:.2f}     | {metrics['2']['f1-score']:.2f}     | {metrics['3']['f1-score']:.2f}     | {metrics['4']['f1-score']:.2f}     | {metrics['macro avg']['f1-score']:.2f}    | {metrics['weighted avg']['f1-score']:.2f}    |
            | Support       | {metrics['0']['support']}       | {metrics['1']['support']}       | {metrics['2']['support']}       | {metrics['3']['support']}       | {metrics['4']['support']}       | {metrics['macro avg']['support']}      | {metrics['weighted avg']['support']}     |
            
            **Accuracy:** {metrics['accuracy']:.2f}  
            **ROC AUC:** {metrics['roc_auc']:.2f}
            """
            st.markdown(performance_metrics)

    else:
        st.warning("Please upload a dataset first.")

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        
        # Plotting with Seaborn
        st.subheader("Drug Distribution")
        plt.figure(figsize=(12, 5), dpi=200)
        sns.countplot(x='Drug', data=df, palette='viridis')  # Using a colorful palette
        plt.title('Distribution of Drugs', fontsize=16)
        plt.xlabel('Drug', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        st.pyplot(plt)

        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 10), dpi=200)
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap', fontsize=16)
        st.pyplot(plt)

        st.subheader("Pairplot")
        sns.pairplot(df, hue='Drug', palette='viridis')
        st.pyplot(plt)
        
    else:
        st.warning("Please upload a dataset first.")
