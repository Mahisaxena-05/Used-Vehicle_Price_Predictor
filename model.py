import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("car.csv")

print(df.head())
print(df.shape)
print(df.info())
print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFuel types:")
print(df["fuel"].value_counts())

print("\nTransmission:")
print(df["transmission"].value_counts())

print("\nOwner:")
print(df["owner"].value_counts())

print("\nSelling price statistics:")
print(df["selling_price"].describe())
# Remove duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Create vehicle age
df["age"] = 2026 - df["year"]

print("\nVehicle age:")
print(df["age"].describe())


# Extract brand
df["brand"] = df["name"].str.split().str[0]

print("\nVehicle brands:")
print(df["brand"].value_counts().head(15))


# Features and target
X = df[[
    "age",
    "km_driven",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]]

y = df["selling_price"]

# 1. Age vs Selling Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="selling_price")
plt.title("Vehicle Age vs Selling Price")
plt.xlabel("Vehicle Age (Years)")
plt.ylabel("Selling Price")
plt.show()


# 2. KM Driven vs Selling Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price")
plt.title("KM Driven vs Selling Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.show()


# 3. Fuel Type vs Selling Price
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="fuel", y="selling_price")
plt.title("Fuel Type vs Selling Price")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.xticks(rotation=20)
plt.show()
print("\nUnique values:")
print(df.nunique())
# Features and target
X = df[[
    "age",
    "km_driven",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]]

y = df["selling_price"]

# One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=[
        "brand",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ],
    dtype=int
)
# Save feature columns for deployment
feature_columns = X.columns.tolist()

with open("feature_columns.pkl", "wb") as file:
    pickle.dump(feature_columns, file)

print("Feature columns saved successfully!")

print("\nEncoded features:")
print(X.head())

print("\nShape after encoding:")
print(X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestRegressor

# Create model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)
# Log transform the target variable
y_train_log = np.log1p(y_train)

# Create a new Random Forest model
log_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train using log-transformed target
log_model.fit(X_train, y_train_log)

# Predict log prices
y_pred_log = log_model.predict(X_test)

# Convert predictions back to original price scale
y_pred_log_original = np.expm1(y_pred_log)

# Evaluate improved model
mae_log = mean_absolute_error(y_test, y_pred_log_original)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log_original))
r2_log = r2_score(y_test, y_pred_log_original)

print("\nLog-Transformed Model Performance:")
print("MAE:", mae_log)
print("RMSE:", rmse_log)
print("R2 Score:", r2_log)

print("Model training completed!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_log_original, alpha=0.5)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price - Improved Model")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()
import pickle

# Save best model
with open("vehicle_price_model.pkl", "wb") as file:
    pickle.dump(log_model, file)

print("Best model saved successfully!")

print("Model saved successfully!")