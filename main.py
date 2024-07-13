import pandas as pd
import numpy as np
from pandas import DataFrame

df: DataFrame = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\data\all_data_apartments.csv")

df.columns = ["id", "city", "type", "square_meters", "rooms", "floor", "floor_count",
"build_year", "latitude", "longitude", "centre_distance", "poi_count", "school_distance",
"clinic_distance", "post_office_distance", "kinder_garten_distance", "restaurant_distance",
"college_distance", "pharmacy_distance", "ownership", "building_material", "condition",
"has_parking_space", "has_balcony", "has_elevator", "has_security", "has_storage_room",
"price", "date", "year", "month"]

df = df.drop(columns = ["date", "year", "month", "latitude", "longitude"])
print(df.isnull().sum())