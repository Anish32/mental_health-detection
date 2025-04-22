import streamlit as st
import pandas as pd
import pickle

# Load the trained model and encoder
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

# Create the Streamlit app
st.title("Mental Health Risk Prediction")
st.write("Enter the following information to predict the risk of mental health issues:")

# Input fields
family_history = st.selectbox("Family History", ["No", "Yes"])
work_hours = st.slider("Work Hours", 0, 24, 8)
sleep_hours = st.slider("Sleep Hours", 0, 24, 7)
social_interaction = st.slider("Social Interaction", 0, 10, 5)
physical_activity = st.slider("Physical Activity", 0, 10, 3)
stress_level = st.slider("Stress Level", 0, 10, 5)
diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])

# Create a dictionary with the input values
input_data = {
    "Family History": ["No"],
    "Work Hours": [7],
    "Sleep Hours": [8],
    "Social Interaction": [7],
    "Physical Activity": [5],
    "Stress Level": [3],
    "Diet Quality": ["Good"]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Encode the input data
encoded_data = encoder.transform(input_df)

# Make a prediction
prediction = model.predict(encoded_data)[0]

# Display the prediction
if prediction == 1:
    st.error("**High Risk of Mental Health Issues**")
else:
    st.success("**Low Risk of Mental Health Issues**")
