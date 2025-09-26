import numpy as np
import pandas as pd

model = pd.read_csv('/Users/rashmiarya/Desktop/ml_projectsm/smoking_health_data_final.csv')

model

model.dropna(inplace=True)

# Ensure blood_pressure is string and clean
model['blood_pressure'] = model['blood_pressure'].astype(str)
model['blood_pressure'] = model['blood_pressure'].str.replace(r'[\[\]\s]', '', regex=True)

# Split safely
bp_split = model['blood_pressure'].str.split('/', expand=True)

# Drop bad rows
bp_split = bp_split.dropna()
bp_split = bp_split[(bp_split[0].str.isnumeric()) & (bp_split[1].str.isnumeric())]

# Assign back
model = model.loc[bp_split.index]
model['systolic'] = bp_split[0].astype(float)
model['diastolic'] = bp_split[1].astype(float)

# Check range
print(model[['systolic', 'diastolic']].describe())
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

model['age']=le.fit_transform(model['age'])
model['sex']=le.fit_transform(model['sex'])
model['current_smoker']=le.fit_transform(model['current_smoker'])

model.describe()

model

x=model[['age',	'sex',	'current_smoker',		'cigs_per_day',	'chol']]

y_hr = model['heart_rate']
y_systolic = model['systolic']
y_diastolic = model['diastolic']

x

y_systolic 
y_diastolic 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Heart rate
x_train_hr, x_test_hr, y_train_hr, y_test_hr = train_test_split(x, y_hr, test_size=0.2, random_state=42)

# Systolic
x_train_sys, x_test_sys, y_train_sys, y_test_sys = train_test_split(x, y_systolic, test_size=0.2, random_state=42)

# Diastolic
x_train_dia, x_test_dia, y_train_dia, y_test_dia = train_test_split(x, y_diastolic, test_size=0.2, random_state=42)

# Models
hr_model = RandomForestRegressor()
sys_model = RandomForestRegressor()
dia_model = RandomForestRegressor()

hr_model.fit(x_train_hr, y_train_hr)
sys_model.fit(x_train_sys, y_train_sys)
dia_model.fit(x_train_dia, y_train_dia)

import pickle

import os
import os

model_path = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')
with open(model_path, 'rb') as f:
    heart_model = pickle.load(f)


import pickle

with open('hr_model.pkl', 'wb') as f:
    pickle.dump(hr_model, f)

with open('systolic_model.pkl', 'wb') as f:
    pickle.dump(sys_model, f)

with open('diastolic_model.pkl', 'wb') as f:
    pickle.dump(dia_model, f)


# def predict_health(age, sex, cigs_per_day):
#     sex_encoded = le.transform([sex])[0]
#     user_input = np.array([[age, sex_encoded, cigs_per_day]])

input_data = [[43, 1, 1, 3, 200]]  # example input

heart_rate_prediction = hr_model.predict(input_data)[0]
systolic_prediction = sys_model.predict(input_data)[0]
diastolic_prediction = dia_model.predict(input_data)[0]

print("\n--- Prediction ---")
print(f"Predicted Heart Rate: {predicted_hr:.2f} bpm")
print(f"Predicted Blood Pressure: {predicted_sys:.2f} / {predicted_dia:.2f} mmHg")






