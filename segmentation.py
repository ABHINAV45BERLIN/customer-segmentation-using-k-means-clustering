import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("customer_segmentation_kmeans.pkl")
scaler = joblib.load("customer_segmentation_scaler.pkl")

st.title("Customer Segmentation Prediction App")
st.write("Enter customer details to predict their segment:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
Income = st.number_input("Income", min_value=0, max_value=200000,value= 10000)
Recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
Total_spending = st.number_input("Total Spending", min_value=0, max_value=100000, value=5000)
NumWebPurchases	 = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=5)
NumStorePurchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
NumWebVisitsMonth = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=5)


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [Income],
    "Recency": [Recency],
    "Total_spending": [Total_spending],
    "NumWebPurchases": [NumWebPurchases],
    "NumStorePurchases": [NumStorePurchases],
    "NumWebVisitsMonth": [NumWebVisitsMonth]
    


})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):

    cluster = kmeans.predict(input_scaled)[0]
    st.write(f"The predicted customer segment is: Cluster {cluster}")
