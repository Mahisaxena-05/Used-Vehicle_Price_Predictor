from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
with open("vehicle_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from form
    brand = request.form["brand"]
    age = int(request.form["age"])
    km_driven = int(request.form["km_driven"])
    fuel = request.form["fuel"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]
    owner = request.form["owner"]

    # Create input dataframe
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

    # Display result on the same page
    return render_template(
        "index.html",
        prediction=f"₹{prediction:,.0f}"
    )


if __name__ == "__main__":
    app.run(debug=True)