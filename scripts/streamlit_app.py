import streamlit as st

model = st.selectbox(
    "Choose a model to simulate", ["Risk Classifier", "Future Revenue Predictor"]
)
st.write(f"You selected: {model}. This app simulates a prediction using that model.")
