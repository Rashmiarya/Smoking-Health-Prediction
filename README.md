# Smoking Health Prediction

This repository contains a machine learning project focused on predicting health outcomes based on smoking status and related health metrics. The goal is to develop a model that can identify individuals at higher risk due to smoking, aiding in proactive health interventions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview
Smoking is a major public health concern, linked to numerous chronic diseases.  
This project aims to leverage machine learning to predict the likelihood of certain health issues given an individual's smoking habits and other physiological data.

By understanding these predictive relationships, we can potentially:
- Identify high-risk individuals for targeted health programs.
- Enhance public health campaigns with data-driven insights.
- Contribute to personalized medicine approaches.

---

## Dataset
The dataset used in this project comprises various health indicators and smoking status for a cohort of individuals. It includes both categorical and numerical features.

Example features include:
- **Smoking Status:** Smoker, Non-smoker, Ex-smoker
- **Age:** Age of the individual
- **Gender:** Male, Female
- **Blood Pressure:** Systolic and Diastolic
- **Cholesterol Levels:** Total, LDL, HDL
- **Glucose Levels:** Fasting glucose
- **Height and Weight:** For BMI calculation
- **Waist Circumference:** Indicator of health risk
- **Physical Activity:** Self-reported levels
- **Alcohol Consumption:** Frequency/amount
- **Family History:** Of certain diseases
- **Target Variable:** Presence/absence of specific health conditions (heart disease, diabetes, stroke risk, etc.)

> **Note:** The exact columns and descriptions will be detailed in the data dictionary or during the data exploration phase.

---

## Features
The features utilized in this model can be categorized as:

- **Demographic Features:** Age, Gender
- **Lifestyle Features:** Smoking Status, Alcohol Consumption, Physical Activity
- **Biometric Features:** Height, Weight, Waist Circumference
- **Clinical Features:** Blood Pressure, Cholesterol Levels, Glucose Levels, Family History

**Feature Engineering may include:**
- Calculating Body Mass Index (BMI) from height and weight.
- Creating interaction terms between smoking status and other health metrics.
- One-hot encoding for categorical variables.

---

## Methodology
The project follows a standard machine learning pipeline:

1. **Data Collection & Loading**
   - Import dataset (e.g., pandas DataFrame).

2. **Exploratory Data Analysis (EDA)**
   - Statistical summaries
   - Visualization of distributions & relationships
   - Handling missing values and outliers
   - Understanding correlations

3. **Data Preprocessing**
   - Imputation for missing values
   - Encoding categorical variables
   - Scaling numerical features
   - Feature engineering
   - Train-test split

4. **Model Selection & Training**
   - Algorithms tested:
     - Logistic Regression
     - SVM
     - Decision Trees
     - Random Forest
     - Gradient Boosting (XGBoost, LightGBM)
     - KNN
   - Cross-validation for evaluation

5. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - ROC-AUC Curve, Confusion Matrix
   - Feature Importance analysis

6. **Hyperparameter Tuning**
   - GridSearchCV / RandomizedSearchCV

---

## Installation

To set up the project environment:

```bash
# Clone the repository
git clone https://github.com/Rashmiarya/Smoking-Health-Prediction.git
cd Smoking-Health-Prediction

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Once the environment is set up, you can run Jupyter notebooks or Python scripts.

```bash
# Navigate to project directory
cd Smoking-Health-Prediction

# Launch Jupyter Notebook
jupyter notebook
```

Then open and run `Smoking_Health_Prediction.ipynb`.

To run a script directly:

```bash
python src/main_script.py   # Replace with actual script name
```

**Workflow**
- Load `data/raw/health_data.csv`
- Run preprocessing steps
- Train models
- Evaluate models & visualize results

---

## Results

This section will be updated once the analysis is complete. Expected results include:

- Comparison of model performances
- ROC AUC curves and confusion matrices
- Feature importance plots
- Insights into smoking and health risks

---

## Contributing

Contributions are welcome!

**Steps:**

```bash
# Fork the repository

# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes

# Commit changes
git commit -m "Add some feature"

# Push branch
git push origin feature/your-feature-name
```
Then open a Pull Request.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## Contact

For questions or suggestions:

**Rashmi Arya**  
📧 Email: rashmiarya007@gmail.com
