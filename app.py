import streamlit as st
from src.prediction import Insurance_Prediction

st.title("Insurance Premium Prediction")

Age = st.number_input("Enter Age")
Annual_Income_LPA = st.number_input("Annual Income (LPA)")
Policy_Term_Years = st.number_input("Policy Term (Years)")
Sum_Assured_Lakhs = st.number_input("Sum Assured (Lakhs)")

if st.button("Predict Premium"):
    model = Insurance_Prediction()
    result = model.prediction(Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs)

    st.success(f"Predicted Premium: {result}")