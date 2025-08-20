# 🚢 Titanic Survival Prediction App

## Overview

This project is a **machine learning-powered web application** that predicts whether a passenger survived the Titanic disaster based on personal and travel information. Built with **Python, scikit-learn, and Streamlit**, the app provides **interactive data exploration, visualization, real-time prediction, and model performance evaluation**.

The machine learning model uses a **Support Vector Machine (SVM) pipeline** with preprocessing, achieving **82% cross-validation accuracy**, outperforming Logistic Regression and Random Forest.

---

## Features

### 1. Data Exploration

- View **dataset overview**: shape, columns, and data types
- Display **sample rows**
- **Interactive filtering**: filter dataset by any column

### 2. Visualisation

- **Survival by gender**
- **Survival by passenger class**
- **Age distribution histogram** with interactive range slider
- **Correlation heatmap**
- All visualizations are interactive using Streamlit widgets

### 3. Model Prediction

- Input widgets for passenger features (slider, selectbox, number input)
- **Real-time survival prediction**
- Display **prediction confidence/probability**

### 4. Model Performance

- Display **confusion matrix** and **classification report**
- Compare **cross-validation accuracy** for Logistic Regression, Random Forest, and SVM
- Highlight **best-performing SVM model**

---

## Technical Details

- **Dataset**: Titanic dataset from Kaggle
- **Preprocessing**:
  - Fill missing values (`Age` median, drop `Cabin`)
  - Encode categorical features (`Sex`, `Embarked`)
  - Create feature `FamilySize = SibSp + Parch + 1`
  - Scale numeric features using `StandardScaler`
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM, best performing)
- **Pipeline**: `ColumnTransformer` for preprocessing + classifier
- **Model Saving**: Saved using `joblib` for deployment

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/titanic-survival-app.git
cd titanic-survival-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

    - Place train.csv inside the data/ folder.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

# Project Structure

titanic-survival-app/
├── app.py # Streamlit web application
├── data/
│ └── train.csv # Titanic dataset
├── best_svm_pipeline.joblib# Saved trained SVM pipeline
├── requirements.txt # Python dependencies
├── README.md # Project documentation

# Model Evaluation

| Model               | CV Accuracy |
| ------------------- | ----------- |
| Logistic Regression | 0.8020      |
| Random Forest       | 0.8076      |
| SVM (Best)          | 0.8202      |

Confusion Matrix & Metrics are available in the Model Performance tab of the app.

# Streamlit UI Features

-Sidebar navigation for different sections: Data Exploration, Visualisation, Model Prediction, Model Performance
-Interactive filters and sliders for visualizations
-Real-time prediction with user input widgets
-Clear layout and documentation to guide the user

# Notes

-Ensure input columns match the original pipeline (Sex, Pclass, Embarked)
-The model pipeline handles all preprocessing internally—no manual encoding is required
-Suitable for educational purposes or demonstration of machine learning deployment

# References

-Titanic Dataset on Kaggle
-Scikit-learn Pipeline Documentation
-Streamlit Documentation

# Author

-Lasindu Kashmira Wickramarathne
-Email: lasinduwickramarathne5@gmail.com
-Phone: 0753167593
