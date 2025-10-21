# 🫀 Heart Disease Prediction App

## 📋 Overview
This project predicts the likelihood of a person developing **heart disease** based on several health-related attributes.  
The workflow included **data cleaning**, **feature engineering**, **model training**, and **deployment using Streamlit** for real-time prediction.

---

## 🧹 Data Cleaning & Preprocessing
- Removed duplicates and irrelevant columns.  
- Handled missing values and outliers.  
- Encoded categorical variables using **LabelEncoder**.  
- Scaled numerical features for better model performance.  
- Split the dataset into **training (75%)** and **testing (25%)** sets.  
- Performed **EDA** (Exploratory Data Analysis) to identify correlations between features and heart disease occurrence.

---

## 📊 Dataset Description
The dataset contains patient medical information used to assess heart disease risk.

| Column Name | Description |
|--------------|-------------|
| **age** | Age of the patient (in years) |
| **sex** | Gender (1 = male, 0 = female) |
| **cp** | Chest pain type (0–3: typical angina to asymptomatic) |
| **trestbps** | Resting blood pressure (mm Hg) |
| **chol** | Serum cholesterol (mg/dl) |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| **restecg** | Resting electrocardiographic results (0–2) |
| **thalach** | Maximum heart rate achieved |
| **exang** | Exercise-induced angina (1 = yes, 0 = no) |
| **oldpeak** | ST depression induced by exercise relative to rest |
| **slope** | Slope of the peak exercise ST segment (0–2) |
| **ca** | Number of major vessels (0–3) colored by fluoroscopy |
| **thal** | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) |
| **target** | Diagnosis of heart disease (1 = presence, 0 = absence) |

---

## 🤖 Machine Learning Models & Results
Multiple models were trained and evaluated to find the most accurate one:

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | **82.07%** |
| Support Vector Machine (SVM) | **84.24%** |
| Decision Tree | **72.83%** |
| Random Forest | **82.61%** |
| **XGBoost** | 🏆 **85.33%** |

✅ **Best Model:** XGBoost achieved the highest accuracy of **85.33%**, making it the final model used in the application.

---

## 💻 Streamlit Web Application
A **Streamlit app** was developed for real-time prediction.  
Users can input health information, and the app instantly predicts the risk of heart disease.

### Features:
- Clean and user-friendly interface.  
- Visualized probability output using **Plotly Gauge Chart**.  
- Displays health risk category:  
  🟢 Low Risk | 🟡 Moderate Risk | 🔴 High Risk  

---

## 🧠 Technologies Used
- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn, XGBoost**
- **Plotly, Streamlit**
- **Joblib**

---

## 🚀 How to Run the App
```bash
# Clone the repository
git clone https://github.com/Keromedhat/Heart-Disease-prediction-app.git

# Navigate to project folder
cd Heart-Disease-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
