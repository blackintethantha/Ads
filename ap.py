import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("🎯 Sales Prediction")
st.write("Enter your Advertistment spending limits.")

# User Inputs
tv_budget = st.number_input("TV Budget ($)", min_value=0, value=100000)
radio_budget = st.number_input("Radio Budget ($)", min_value=0, value=500000)
newspaper_budget = st.number_input("Newspaper Budget ($)", min_value=0, value=250000)

# Compute Total Budget (same as during model training)
total_budget = tv_budget + radio_budget + newspaper_budget

# Prepare input data for prediction
user_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget, total_budget]],
                         columns=["TV", "Radio", "Newspaper", "Total_Budget"])

# Predict Sales
prediction = model.predict(user_data)

# Show result when "Predict" button is clicked
if st.button("Predict"):
    st.success(f"📈 Predicted Sales: {prediction[0]:,.2f} units")
