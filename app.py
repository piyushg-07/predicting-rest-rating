import streamlit as st
import numpy as np
from sklearn.preprocessing import  StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("scaler.pkl")

st.title("Restaurant Rating Prediction")


st.caption("This app helps  you predict the rating of a restaurant based on its features.")

st.divider()

averagecost = st.number_input("Average Cost for two", min_value=50, max_value=10000, value=1000, step=200)

tablebooking = st.selectbox("Restaurant has Table Booking", ["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has Online Delivery?", ["Yes", "No"])


princerange = st.selectbox("What is Price Range(1 cheapest, 4 Most Expensive)", ["1", "2", "3", "4"])

predictbutton = st.button("Predict the Rating")
st.divider()
model = joblib.load("model.pkl")

bookingstatus = 1  if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, princerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

# avegarecost_scaled = scaler.transform(averagecost)

# bookingstatus = scaler.fit_transform(bookingstatus)

# deliverystatus = scaler.fit_transform(deliverystatus)

# princerange = scaler.fit_transform(princerange)

if predictbutton:
    st.snow()
    
    prediction = model.predict(X)
    
    st.write(prediction)
    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif  prediction < 4.0:
        st.write("Good") 
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent") 
            