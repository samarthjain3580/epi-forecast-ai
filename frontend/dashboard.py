import streamlit as st
import requests

st.title("Epidemic Forecast AI")

st.write("Predict future epidemic cases using LSTM")

if st.button("Get Prediction"):
    response = requests.get("http://127.0.0.1:5000/predict")
    data = response.json()

    st.write("Predicted cases:")
    st.write(data["predicted_cases"])