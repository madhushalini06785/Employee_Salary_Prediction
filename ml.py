import pandas as pd
data=pd.read_csv(r"C:\Users\hii\Desktop\salary_prediction_project\dataset.csv")
data
data.shape
data.isna()
data.isna().sum()
data
data.drop(columns=['Hire_Date'],inplace=True)
data
#delete the rows who are resigned
data.drop(data[data['Resigned'] == True].index, inplace=True)
data
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['Department']=encoder.fit_transform(data['Department'])
data['Gender']=encoder.fit_transform(data['Gender'])
data['Job_Title']=encoder.fit_transform(data['Job_Title'])
data['Education_Level']=encoder.fit_transform(data['Education_Level'])
data['Resigned']=encoder.fit_transform(data['Resigned'])
data
#input and output
x=data.drop(columns=['Monthly_Salary'])
y=data['Monthly_Salary']
x
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred =model.predict(X_test)
print("\nRandom Forest Regressor:")
print("MSE:", mean_squared_error(y_test,pred))
print("R² Score:", r2_score(y_test,pred))

import pickle
# Save the model
with open(r"C:\Users\hii\Desktop\salary_prediction_project\salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save column names
with open(r"C:\Users\hii\Desktop\salary_prediction_project\model_columns.pkl", "wb") as f:
    pickle.dump(x.columns.tolist(), f)

print("✅ Model and columns saved successfully.")
