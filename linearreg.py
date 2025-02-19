import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pickle import load

# Load dataset
data = pd.read_csv("Student_Performance.csv")
X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
Y = data['Performance Index']

# Split data for scaling and modeling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load trained Linear Regression model
loaded_model = load(open('lr', 'rb'))

def performance_prediction(input_data):
    """Predicts student performance based on input features."""
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return round(prediction[0], 2)

def main():
    st.title('Student Performance Prediction using Linear Regression')
    
    # Collect user inputs
    hours_studied = st.number_input('Hours Studied', min_value=0.0, format="%.1f")
    previous_scores = st.number_input('Previous Scores', min_value=0.0, format="%.1f")
    extracurricular = st.number_input('Extracurricular Activities', min_value=0)
    sleep_hours = st.number_input('Sleep Hours', min_value=0.0, format="%.1f")
    sample_papers = st.number_input('Sample Question Papers Practiced', min_value=0)
    
    prediction = ''
    if st.button('Predict Performance Index'):
        prediction = performance_prediction([hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers])
        st.success(f'Predicted Performance Index: {prediction}')
    
if __name__ == '__main__':
    main()