# app.py
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("salary_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Age": float(request.form['Age']),
        "Years_At_Company": float(request.form['Years_At_Company']),
        "Education_Level": request.form['Education_Level'],
        "Department": request.form['Department'],
        "Performance_Score": int(request.form['Performance_Score']),
        "Work_Hours_Per_Week": float(request.form['Work_Hours_Per_Week']),
        "Projects_Handled": int(request.form['Projects_Handled']),
        "Overtime_Hours": float(request.form['Overtime_Hours']),
        "Sick_Days": int(request.form['Sick_Days']),
        "Remote_Work_Frequency": int(request.form['Remote_Work_Frequency']),
        "Team_Size": int(request.form['Team_Size']),
        "Training_Hours": float(request.form['Training_Hours']),
        "Promotions": int(request.form['Promotions']),
        "Employee_Satisfaction_Score": float(request.form['Employee_Satisfaction_Score']),
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    prediction = model.predict(df)[0]

    return render_template('index.html', prediction_text=f"Estimated Monthly Salary: ₹{round(prediction, 2)}")

if __name__ == '__main__':
    app.run(debug=True)
