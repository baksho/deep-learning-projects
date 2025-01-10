import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
model_path = r"C:\Users\Suvinava Basak\Documents\pythonScripts\deep_learning"
model_path = model_path + r"\house_price_prediction_model.keras"
model = load_model(model_path)

preprocessor_path = r"C:\Users\Suvinava Basak\Documents\pythonScripts\deep_learning"
preprocessor_path = preprocessor_path + r"\preprocessor.pkl"
preprocessor = joblib.load(preprocessor_path)


# Function to take user input
def get_user_input():
    print("\nPlease enter the details of the house:")
    suburb = input("Suburb (e.g., PERTH): ").strip().upper()
    nearest_stn = input("Nearest Station: ").strip().upper()
    nearest_sch = input("Nearest School: ").strip().upper()
    bedrooms = int(input("Number of Bedrooms: "))
    bathrooms = int(input("Number of Bathrooms: "))
    garage = float(input("Number of Garage Spaces (can be 0 if none): "))
    land_area = int(input("Land Area (in square meters): "))
    floor_area = int(input("Floor Area (in square meters): "))
    build_year = int(input("Build Year: "))
    cbd_dist = int(input("Distance to CBD (in km): "))
    nearest_stn_dist = int(input("Distance to Nearest Station (in km): "))
    nearest_sch_dist = float(input("Distance to Nearest School (in km): "))
    longitude = float(input("Longitude: "))
    latitude = float(input("Latitude: "))
    postcode = int(input("Postcode: "))
    year_sold = int(input("Year Sold: "))
    month_sold = int(input("Month Sold (Give number of month): "))

    # Create a DataFrame for user input
    user_data = pd.DataFrame(
        {
            "SUBURB": [suburb],
            "NEAREST_STN": [nearest_stn],
            "NEAREST_SCH": [nearest_sch],
            "BEDROOMS": [bedrooms],
            "BATHROOMS": [bathrooms],
            "GARAGE": [garage],
            "LAND_AREA": [land_area],
            "FLOOR_AREA": [floor_area],
            "BUILD_YEAR": [build_year],
            "CBD_DIST": [cbd_dist],
            "NEAREST_STN_DIST": [nearest_stn_dist],
            "NEAREST_SCH_DIST": [nearest_sch_dist],
            "LONGITUDE": [longitude],
            "LATITUDE": [latitude],
            "POSTCODE": [postcode],
            "YEAR_SOLD": [year_sold],
            "MONTH_SOLD": [month_sold],
        }
    )
    return user_data


# Function to predict the price
def predict_price(user_input):
    # Preprocess the input data
    user_input_transformed = preprocessor.transform(user_input)

    # Predict the price
    predicted_price = model.predict(user_input_transformed)
    return predicted_price[0][0]


if __name__ == "__main__":
    # Get user input
    user_input = get_user_input()

    # Predict the price
    predicted_price = predict_price(user_input)

    # Display the predicted price
    print("\nPredicted House Price:")
    print(f"${predicted_price:,.2f}")
