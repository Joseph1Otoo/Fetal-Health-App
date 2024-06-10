import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgb_fetal_model.json')

# Feature names
feature_names = [
    'baseline value', 'accelerations', 'fetal_movement',
    'uterine_contractions', 'light_decelerations', 'severe_decelerations',
    'prolongued_decelerations', 'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
    'histogram_median', 'histogram_variance', 'histogram_tendency'
]

# Define Streamlit app
st.title('Fetal Health State Prediction')
st.write('Enter the fetal variables to predict the outcome')

# Define the input fields
inputs = []
for feature in feature_names:
    inputs.append(st.number_input(feature, value=0.0))

if st.button('Predict'):
    input_data = np.array(inputs).reshape(1, -1)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    prediction = model.predict(input_data)
    st.write(f'Predicted Fetal Health State: {prediction[0]}')
