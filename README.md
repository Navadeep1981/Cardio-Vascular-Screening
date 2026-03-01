# 🫀 Cardiovascular Risk Prediction System

This project predicts whether a patient is at risk of progressing to a critical cardiovascular condition using Machine Learning.

It uses synthetic healthcare data, multiple ML models, and a Streamlit web application for real-time prediction.

---

## 📌 What This Project Does

- Takes patient clinical data as input
- Predicts cardiovascular risk
- Shows probability of critical condition
- Asks lifestyle-related questions
- Gives personalized recommendations
- Saves patient records with date and time
- Compares multiple ML models

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CTGAN (for synthetic data generation)
- Streamlit (for web application)
- Matplotlib (for model comparison graphs)
- Joblib (for saving models)

---

## 📊 Machine Learning Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost (Final Selected Model)

XGBoost gave the best accuracy and was used in the final web app.

---

## 🧠 Feature Engineering

We created one important medical feature:

Pulse Pressure = Systolic BP − Diastolic BP

This helped improve model performance.

---

## 🚀 How to Run the Project
streamlit run app.py


The application will open in your browser.

---

## 💻 Web Application Features

- Patient ID input
- Clinical data input
- Risk prediction with probability
- Lifestyle questionnaire
- Health recommendations
- Patient data saved automatically
- Model comparison dashboard

---

 
