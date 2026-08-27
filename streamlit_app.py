import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Used Vehicle Price Predictor",
    
    layout="centered"
)

# Load model
with open("vehicle_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


st.title("🚗 Used Vehicle Price Predictor")
st.write("Enter the vehicle details to estimate its selling price.")


# Input fields
brand = st.selectbox(
    "Vehicle Brand",
    [
        "Maruti", "Hyundai", "Mahindra", "Tata", "Ford",
        "Honda", "Toyota", "Chevrolet", "Renault",
        "Volkswagen", "Nissan", "Skoda", "Fiat",
        "Audi", "Datsun"
    ]
)

age = st.number_input(
    "Vehicle Age (Years)",
    min_value=1,
    max_value=50,
    value=10
)

km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=1000000,
    value=50000
)

fuel = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)


# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "age": [age],
        "km_driven": [km_driven],
        "brand": [brand],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "brand",
            "fuel",
            "seller_type",
            "transmission",
            "owner"
        ],
        dtype=int
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹{prediction:,.0f}")