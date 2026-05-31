# CodeAlpha_UCI_Credit_Scoring_Model
# 💳 Credit Scoring Model

A Machine Learning-based Credit Scoring System that predicts the probability of customer default based on financial and demographic information.

## 📌 Project Overview

This project uses machine learning techniques to assess credit risk and classify customers as:

- ✅ Low Risk (Creditworthy)
- ⚠️ High Risk (Likely to Default)

The model is trained on the UCI Credit Card Default Dataset and deployed using Streamlit for an interactive user interface.

---

## 🚀 Features

- Credit risk prediction
- Probability of default estimation
- Interactive Streamlit dashboard
- Feature engineering for improved performance
- Machine Learning pipeline with preprocessing
- Model persistence using Joblib

---

## 📂 Project Structure

```text
Credit-Scoring-Model/
│
├── data/
│   └── UCI_Credit_Card.csv
│
├── models/
│   └── credit_pipeline.pkl
│
├── src/
│   └── train.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib

---

## 📊 Dataset

Dataset: UCI Credit Card Default Dataset

Features include:

- Credit Limit
- Gender
- Education
- Marriage Status
- Age
- Repayment History
- Bill Amounts
- Payment Amounts

Target Variable:

- `1` → Default
- `0` → No Default

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/CodeAlpha_UCI_Credit_Scoring_Model.git
cd CodeAlpha_UCI_Credit_Scoring_Model
```

### Create Virtual Environment

```bash
python -m venv env
```

Activate Environment:

Windows:

```bash
env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python src/train.py
```

This will:

- Load dataset
- Perform preprocessing
- Train model
- Save pipeline to:

```text
models/credit_pipeline.pkl
```

---

## ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

Application will open in your browser:

```text
http://localhost:8501
```

---

## 📈 Model Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Feature Scaling & Encoding
5. Model Training
6. Model Evaluation
7. Prediction
8. Deployment

---

## 📊 Sample Output

Prediction Result:

```text
Default Probability: 18.5%

Risk Level:
✅ Low Risk
```

---

## 🔮 Future Improvements

- SHAP Explainability
- Feature Importance Visualization
- REST API using FastAPI
- Docker Deployment
- Cloud Deployment (AWS/Azure)

---

## 👩‍💻 Author

Nandini Tharigoppula

GitHub:
https://github.com/Nandinitharigoppulla

---

## 📄 License

This project is developed for educational and learning purposes.