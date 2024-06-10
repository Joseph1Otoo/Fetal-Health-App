import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgb_fetal_model.json')

# Define Streamlit app
st.title('Fetal Health State Prediction')
st.write('Enter the fetal variables to predict the outcome')

# Define the input fields
inputs = []
for i in range(21):  # Assuming there are 21 features
    inputs.append(st.number_input(f'Feature {i+1}', value=0.0))

if st.button('Predict'):
    input_data = np.array(inputs).reshape(1, -1)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    prediction = model.predict(input_data)
    st.write(f'Predicted Fetal Health State: {prediction[0]}')

if __name__ == '__main__':
    st.run()
