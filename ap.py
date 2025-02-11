import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🎯 Advertising Sales Prediction App")
st.write("Enter ad spending details to predict sales.")

# User Inputs
def get_budget_input(label, default):
    return st.number_input(label, min_value=0, value=default)

tv_budget = get_budget_input("TV Budget ($)", 100)
radio_budget = get_budget_input("Radio Budget ($)", 50)
newspaper_budget = get_budget_input("Newspaper Budget ($)", 25)

# Compute Total Budget
total_budget = sum([tv_budget, radio_budget, newspaper_budget])

# Prepare input data for prediction
user_data = pd.DataFrame({
    "TV": [tv_budget],
    "Radio": [radio_budget],
    "Newspaper": [newspaper_budget],
    "Total_Budget": [total_budget]
})

# Predict Sales
prediction = model.predict(user_data)[0]

# Show result when "Predict" button is clicked
if st.button("Predict"):
    st.success(f"📈 Predicted Sales: {prediction:,.2f} units")
