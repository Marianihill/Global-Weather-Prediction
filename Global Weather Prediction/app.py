import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained ANN model
model = load_model("GlobalWeather.h5")  # .h5 file

# Optional: Load scaler if used
# scaler = joblib.load("scaler.pkl")

# Inject cream background style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff8e1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input feature list
features = [
    'humidity',
    'pressure_mb',
    'wind_kph',
    'air_quality_Carbon_Monoxide',
    'air_quality_PM2.5',
    'air_quality_PM10'
]

# Title and instructions
st.title("Temperature Prediction using ANN")
st.markdown("Enter environmental and air quality data to predict temperature in °C")

# Input fields
user_inputs = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    user_inputs.append(value)

# Prediction logic
if st.button("Predict Temperature (°C)"):
    input_array = np.array(user_inputs).reshape(1, -1)

    # Optional: apply scaler if used
    # input_array = scaler.transform(input_array)

    prediction = model.predict(input_array)
    st.success(f"Predicted Temperature: {prediction[0][0]:.2f} °C")
