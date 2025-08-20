# ====================== IMPORT LIBRARIES ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Titanic Survival Prediction",
                   layout="wide",
                   page_icon="🚢")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
This app allows you to explore the Titanic dataset, visualize key statistics, and predict passenger survival using a trained **SVM model with preprocessing pipeline**.  
The model is based on features like Age, Sex, Passenger Class, Fare, Family Size, and Embarked port.
""")

# ====================== LOAD DATA AND MODEL ======================
@st.cache_data
def load_data():
    df = pd.read_csv("./data/train.csv")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

@st.cache_resource
def load_model():
    model = joblib.load("./best_svm_pipeline.joblib")
    return model

df = load_data()
model = load_model()

# ====================== SIDEBAR NAVIGATION ======================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Data Exploration", "Visualisation", "Model Prediction", "Model Performance"])

# ====================== DATA EXPLORATION ======================
if menu == "Data Exploration":
    st.header("Data Exploration")

    st.subheader("Dataset Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Data types:")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Interactive Data Filtering")
    column_to_filter = st.selectbox("Select column to filter:", df.columns)
    unique_values = df[column_to_filter].unique()
    selected_values = st.multiselect(f"Select values for {column_to_filter}:", unique_values, default=unique_values)
    filtered_data = df[df[column_to_filter].isin(selected_values)]
    st.write(f"Filtered dataset ({filtered_data.shape[0]} rows):")
    st.dataframe(filtered_data)

# ====================== VISUALISATION ======================
elif menu == "Visualisation":
    st.header("Data Visualisation")

    st.subheader("1. Survival by Sex")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Sex", hue="Survived", ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Survival by Passenger Class")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Age Distribution")
    age_range = st.slider("Select Age Range:", int(df["Age"].min()), int(df["Age"].max()), (0, 80))
    fig3, ax3 = plt.subplots()
    sns.histplot(df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]["Age"], kde=True, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Optional: Interactive Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        fig4, ax4 = plt.subplots(figsize=(10,6))
        numeric_df = df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

# ====================== MODEL PREDICTION ======================
elif menu == "Model Prediction":
    st.header("Predict Titanic Survival")

    st.subheader("Enter Passenger Details")

    try:
        pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3])
        sex = st.selectbox("Sex:", ["male", "female"])
        age = st.number_input("Age:", min_value=0.0, max_value=100.0, value=30.0)
        sibsp = st.number_input("Number of Siblings/Spouses aboard:", min_value=0, max_value=10, value=0)
        parch = st.number_input("Number of Parents/Children aboard:", min_value=0, max_value=10, value=0)
        fare = st.number_input("Ticket Fare:", min_value=0.0, max_value=500.0, value=32.2)
        embarked = st.selectbox("Port of Embarkation:", ["C", "Q", "S"])

        family_size = sibsp + parch + 1

        input_df = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
            "FamilySize": [family_size]
        })

        if st.button("Predict Survival"):
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][prediction]

            st.success(f"Prediction: {'Survived' if prediction==1 else 'Did Not Survive'}")
            st.info(f"Prediction Confidence: {prediction_proba*100:.2f}%")

    except Exception as e:
        st.error(f"Error in input: {e}")

# ====================== MODEL PERFORMANCE ======================
elif menu == "Model Performance":
    st.header("Model Evaluation")

    st.subheader("Confusion Matrix & Metrics")
    X = df.drop(columns=["PassengerId","Name","Ticket","Cabin","Survived"], errors='ignore')
    y = df["Survived"]
    preds = model.predict(X)

    cm = confusion_matrix(y, preds)
    st.write("Confusion Matrix:")
    st.write(cm)

    st.subheader("Classification Report")
    report = classification_report(y, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Model Comparison (Cross-validation Accuracy)")
    st.write("""
    | Model | CV Accuracy |
    |-------|-------------|
    | Logistic Regression | 0.8020 |
    | Random Forest | 0.8076 |
    | SVM (Current Best) | 0.8202 |
    """)

    st.info("SVM currently gives the highest cross-validation accuracy on this dataset.")
