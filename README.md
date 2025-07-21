# Employee_Salary_Prediction
A machine learning project that predicts employee salaries based on features like experience, education, job role,etc. Trained using Random Forest on structured data to estimate monthly salary.

# Project Overview
Goal: Predict employee salary using features such as age, education, department, experience, work hours, satisfaction score, and more.

# ML Models Used
Linear Regression
K-Nearest Neighbors (KNN)
XGBoost
Random Forest Regressor (Best Performing)

# Why Random Forest?
After evaluating all models, Random Forest Regressor gave the best results:
R² Score: 1.0
MSE: 0.0
This indicates perfect prediction on the test data and makes it the best choice.

# Dataset
Name: dataset.csv
Target Variable: Monthly_Salary (Manually mapped to realistic salary ranges: ₹15,000 – ₹50,000)

# Features Include:
Age, Gender, Education Level, Department, Job Title
Years at Company, Performance Score, Work Hours
Overtime, Sick Days, Remote Frequency, Team Size
Training Hours, Promotions, Satisfaction Score

# Features of the Web App
HTML form with dropdowns/placeholders for easy input
Real-time prediction using trained model (salary_model.pkl)
Flask handles backend prediction logic
One-hot encoding handled using saved columns (model_columns.pkl)
Styled with CSS for user-friendly UI

# Tech Stack
Python
Flask
Pandas, NumPy
Scikit-learn, XGBoost
HTML5, CSS3

# How to Run Locally
# Clone the repo
git clone https://github.com/madhushalini06785/Employee_Salary_Prediction
cd Employee_Salary_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
Visit the app at: http://127.0.0.1:5000

📁 Folder Structure
2.salary_prediction_project/
├── app.py                       # Flask app
├── ml.py                        # ML training script
├── dataset.csv                  # Dataset
├── salary_model.pkl             # Trained model
├── model_columns.pkl            # Column info for prediction
├── requirements.txt             # All dependencies
├── static/
│   └── style.css                # CSS styling
└── templates/
    └── index.html               # Web interface

# Sample Prediction Output
Estimated Monthly Salary: ₹42,000.00

📬 Contact
Name: Madhu Shalini Kanneboina
Email:kanneboina.madhushalini@gmail.com.com
LinkedIn: linkedin.com/in/your-profile
