from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
hr_model = joblib.load('hr_model.pkl')
sys_model = joblib.load('systolic_model.pkl')
dia_model = joblib.load('diastolic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Your input form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # 0 = female, 1 = male
        is_smoker = int(request.form['is_smoker'])  # 0 or 1
        cigs_per_day = int(request.form['cigs_per_day'])
        chol = int(request.form['chol'])

        # Create input array for prediction
        input_data = np.array([[age, sex, is_smoker, cigs_per_day, chol]])

        # Make predictions
        predicted_hr = hr_model.predict(input_data)[0]
        predicted_sys = sys_model.predict(input_data)[0]
        predicted_dia = dia_model.predict(input_data)[0]

        return render_template(
    "result.html",
    predicted_hr=predicted_hr,
    predicted_sys=predicted_sys,
    predicted_dia=predicted_dia
)

    except Exception as e:
        return f"❌ Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
