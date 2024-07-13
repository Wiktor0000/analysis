from typing import Any

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Wczytywanie danych
df: DataFrame = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\data\all_data_apartments.csv")

#Zmiana nazw kolumn
df.columns = ["id", "city", "type", "square_meters", "rooms", "floor", "floor_count",
"build_year", "latitude", "longitude", "centre_distance", "poi_count", "school_distance",
"clinic_distance", "post_office_distance", "kinder_garten_distance", "restaurant_distance",
"college_distance", "pharmacy_distance", "ownership", "building_material", "condition",
"has_parking_space", "has_balcony", "has_elevator", "has_security", "has_storage_room",
"price", "date", "year", "month"]

#Usunięcie niepotrzebnych kolumn
df = df.drop(columns = ["date", "year", "month", "latitude", "longitude"])


#Uzupełnienie brakujących wartości, najczęściej występującą wartością.
mode_value = df["type"].mode()[0]
df["type"] = df["type"].fillna(mode_value)

columns_to_fill = ["floor", "floor_count", "build_year", "school_distance", "clinic_distance",
                   "post_office_distance", "kinder_garten_distance", "restaurant_distance",
                   "college_distance", "pharmacy_distance"]
for col in columns_to_fill:
    average_value = df[col].mean()
    df[col] = df[col].fillna(average_value)

lack_value = ["building_material", "condition", "has_elevator"]

for lack in lack_value:
    df[lack] = df[lack].fillna("Brak danych")

#Zmiana danych ze zmiennoprzecinkowych, na całkowite
data_type_mapping = {
    "rooms": int,
    "floor": int,
    "floor_count": int,
    "poi_count":int,
    "build_year":int
}
for col, dtype in data_type_mapping.items():
    df[col] = df[col].astype(dtype, errors = "ignore")

#Regresja liniowa
x = df["square_meters"].values
y = df["price"].values
x_mean = np.mean(x)
y_mean = np.mean(y)

beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta_0 = y_mean - (beta_1 * x_mean)

#Wyznaczanie prostej regresji - metoda najmniejszych kwadratów (MNK)
prosta = (beta_1 * x) + beta_0