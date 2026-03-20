# 🚀 AutoML Streamlit Web App

## 📌 Project Overview
This project is a **Streamlit-based AutoML application** that allows users to upload a dataset and automatically train machine learning models without writing code.

The app intelligently detects whether the problem is **classification or regression**, preprocesses the data, trains multiple models, tunes hyperparameters, and displays the best results.

---

## ⚙️ Features
- Upload CSV dataset  
- Select target column dynamically  
- Automatic task detection (Classification / Regression)  
- Data preprocessing:
  - Missing value handling  
  - Encoding categorical features  
  - Feature scaling  
- Train multiple ML models  
- Hyperparameter tuning  
- Model comparison  
- Visual evaluation metrics  
- Predict using trained model  

---

## 🧠 Machine Learning Models

### Classification
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

### Regression
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

---

## 📂 Project Structure
```text
Ml_project/
│
├── app.py
├── requirements.txt
├── README.md
│
└── src/
    ├── preprocessing.py
    ├── model_training.py
    ├── tuning.py
    └── evaluation.py
```

---

## ⚡ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
```

### 3. Activate environment
**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application
```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 📊 How to Use
1. Upload a CSV dataset  
2. Select the target column  
3. App detects problem type automatically  
4. Click "Train Model"  
5. View performance metrics  
6. Enter values and get predictions  

---


## 👥 Contributors
- Jegan R K 
- Gnaneswaran B S  

---
