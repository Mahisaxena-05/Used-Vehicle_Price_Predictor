# Used Vehicle Price Predictor

An end-to-end Machine Learning web application that predicts the selling price of a used vehicle based on its age, kilometers driven, fuel type, seller type, transmission, and owner details.

##  Project Overview

The project uses the **Vehicle Dataset from CarDekho** to train a regression model for used vehicle price prediction.

The trained Machine Learning model is integrated with a **Flask web application**, allowing users to enter vehicle details and get an estimated selling price instantly.

##  Features

* Predict used vehicle selling price
* Vehicle age-based prediction
* Kilometer-driven based prediction
* Supports multiple fuel types
* Supports different seller types
* Manual and automatic transmission
* Different owner categories
* Simple and user-friendly web interface
* Machine Learning based price estimation

##  Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-Learn**
* **Matplotlib**
* **Seaborn**
* **Flask**
* **Pickle**
* **HTML & CSS**

##  Machine Learning

A **Random Forest Regressor** is used for price prediction.

The dataset was cleaned by:

1. Checking missing values
2. Removing duplicate rows
3. Creating vehicle age from the manufacturing year
4. Extracting vehicle brand from the vehicle name
5. Applying One-Hot Encoding to categorical features
6. Splitting the data into training and testing sets
7. Training the Random Forest model

A log-transformed target model was also evaluated to improve prediction performance.

##  Model Performance

### Log-Transformed Random Forest Model

* **MAE:** 148,671.69
* **RMSE:** 351,947.69
* **R² Score:** 0.6155

The model achieved an **R² score of approximately 61.5%** on the test data.

##  Project Structure

```text
Used-Vehicle_Price_Predictor/
│
├── app.py
├── model.py
├── car.csv
├── vehicle_price_model.pkl
├── feature_columns.pkl
├── requirements.txt
├── .gitignore
│
└── templates/
    └── index.html
```

##  Run Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

##  Project Goal

The goal of this project is to demonstrate an **end-to-end Machine Learning workflow**, from data preprocessing and model training to deploying the trained model as a functional web application.

## Dataset Link:
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

##  Author

**Mahi Saxena**
