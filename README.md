# AI-Powered Alternate Credit Scoring Platform

An AI-driven credit risk intelligence system that predicts Probability of Default (PD) using machine learning and alternate data sources to enhance financial inclusion and reduce NPAs.

## 📌 Problem Statement

Traditional credit scoring systems rely heavily on past bureau data, excluding thin-file and credit-invisible users.  
This project builds a dynamic, explainable, and scalable AI-based risk assessment platform that:

- Predicts Probability of Default (PD)
- Generates standardized credit scores (300–850)
- Integrates alternate behavioral data
- Detects fraud & anomalies
- Enables early warning monitoring

### 🚀 Solution

We propose an AI-based Lending Risk Intelligence Platform that enhances credit risk assessment using machine learning and alternate data sources.

🔹 1. Probability of Default (PD) Prediction
Train supervised ML models on historical repayment data.
Predict the likelihood that a borrower will default.
Use advanced models like XGBoost for high accuracy.

🔹 2. Credit Score Generation (300–850 Scale)
Convert the predicted PD into a standardized credit score.
Higher score → Lower risk
Lower score → Higher risk
Enables easy integration into banking systems.

🔹 3. Alternate & Behavioral Data Integration
To improve inclusion of thin-file users, the system incorporates:
UPI transaction patterns
Utility bill payments
GST data (for MSMEs)
Spending & repayment behavior
This allows scoring even without traditional bureau history.

🔹 4. Feature Engineering & Risk Modeling
Income stability index
Debt-to-income ratios
Transaction consistency metrics
Outlier detection (IQR, Z-score)
Data preprocessing & scaling

🔹 5. Decision Engine
Based on risk score:
Approve (Low Risk)
Review (Medium Risk)
Reject (High Risk)

🔹 6. Fraud & Risk Monitoring
Real-time anomaly detection
Bias monitoring
Identity verification checks

🔹 7. Post-Loan Monitoring & Early Warning System
Continuous behavioral tracking
Early stress signal detection
Automated alerts before default

### 🛠️ Technologies Used

- Python
- Scikit-learn
- XGBoost
- SHAP
- Pandas & NumPy
- SQL
- Streamlit (for UI)




  
