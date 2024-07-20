from typing import Any

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Wczytywanie danych
df: DataFrame = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\data\all_data_apartments.csv")

# Zmiana nazw kolumn
df.columns = ["id", "city", "type", "square_meters", "rooms", "floor", "floor_count",
"build_year", "latitude", "longitude", "centre_distance", "poi_count", "school_distance",
"clinic_distance", "post_office_distance", "kinder_garten_distance", "restaurant_distance",
"college_distance", "pharmacy_distance", "ownership", "building_material", "condition",
"has_parking_space", "has_balcony", "has_elevator", "has_security", "has_storage_room",
"price", "date", "year", "month"]

# Usunięcie niepotrzebnych kolumn
df = df.drop(columns = ["date", "year", "month", "latitude", "longitude"])

# Uzupełnienie brakujących wartości, najczęściej występującą wartością
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

# Zmiana danych ze zmiennoprzecinkowych, na całkowite
data_type_mapping = {
    "rooms": int,
    "floor": int,
    "floor_count": int,
    "poi_count":int,
    "build_year":int
}
for col, dtype in data_type_mapping.items():
    df[col] = df[col].astype(dtype, errors = "ignore")

average_price = df["price"].mean()
print(average_price)

# Współczynnik korelacji Pearsona
correlation_data = df[["price", "square_meters", "rooms"]]
correlation_matrix = correlation_data.corr()

print("Macierz korelacji: ")
print(correlation_matrix)

# Wykres macierzy
plt.figure(figsize = (10, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", center = 0)
plt.title("Macierz korelacji")
plt.show()

# Regresja liniowa - cena vs powierzchnia
def price_and_square_meters(df):
    X = df[["square_meters"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = (train_test_split
        (X, y, test_size = 0.2, random_state = 42))
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Wykres danych
    plt.figure(figsize = (10, 6))
    plt.scatter(X_test, y_test, color = "blue", label = "Dane rzeczywiste", s = 5)
    plt.plot(X_test, y_pred, color = "red", linewidth = 2, label = "Prosta regresji")
    plt.xlabel("Powierzchnia")
    plt.ylabel("Cena")
    plt.title("Regresja liniowa - cena vs powierzchnia")
    plt.legend()
    (plt.gca().get_yaxis().set_major_formatter
        (plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))))
    plt.show()

    return model, X_test, y_test, y_pred, mse, r2

model, X_test, y_test, y_pred, mse, r2 = price_and_square_meters(df)
print(f"Błąd średniokwadratowy (MSE): {mse}")
print(f"Współczynnik determinacji (R^2): {r2}")

# Regresja liniowa - cena vs liczba pokoi
def price_and_rooms(df):
    X = df[["rooms"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Wykres danych
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Dane rzeczywiste', s=10)
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prosta regresji')
    plt.xlabel('Liczba pokoi')
    plt.ylabel('Cena')
    plt.title('Regresja liniowa - Metoda najmniejszych kwadratów')
    plt.legend()
    (plt.gca().get_yaxis().set_major_formatter
        (plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))))
    plt.show()

    return model, X_test, y_test, y_pred, mse, r2
model, X_test, y_test, y_pred, mse, r2 = price_and_rooms(df)
print(f"Błąd średniokwadratowy (MSE): {mse}")
print(f"Współczynnik determinacji (R^2): {r2}")


# Regresja liniowa - cena vs odległość od centrum
def price_and_multi(df):
    X = df[["square_meters", "rooms", "floor",
            "centre_distance"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = (train_test_split
                (X, y, test_size = 0.2, random_State = 42 ))
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


# Rozkład zmiennych.
# Wykres rozkładu powierzchni (metry kwadratowe)
plt.figure(figsize=(10, 6))
sns.histplot(df['square_meters'], kde=True, bins=30)
plt.xlabel('Powierzchnia (metry kwadratowe)')
plt.ylabel('Liczba mieszkań')
plt.title('Rozkład powierzchni mieszkań')
plt.show()

#Wykres rozkładu - liczba pokoi.
plt.figure(figsize=(10, 6))
sns.histplot(df["rooms"], kde = True, bins = 30)
plt.xlabel("Ilość pokoi w mieszkaniu")
plt.ylabel("Liczba mieszkań")
plt.title("Rozkład zmiennych - liczba pokoi")
plt.show()

#Wykres rozkładu - punkty zainteresowania (poi)
plt.figure(figsize = (10, 6))
sns.histplot(df["poi_count"], kde = True, bins = 30)
plt.xlabel("Liczba poi w okolicy")
plt.ylabel(" ")
plt.title("Punkty zainteresowania (poi)")
plt.xticks(ticks = np.arange(0, df["poi_count"].max() + 1, 10))
plt.show()

#Wykres rozkładu - odległość od centrum
plt.figure(figsize = (10, 6))
sns.histplot(df["centre_distance"], kde = True, bins = 30)
plt.xlabel("Odległość od centrum (km)")
plt.ylabel("Liczba mieszkań")
plt.title("Odległość od centrum")
plt.yticks(ticks = np.arange(0, 17000, 5000))
plt.show()

count_elevator = df["has_elevator"].value_counts()
count_parking_space = df["has_parking_space"].value_counts()
count_security = df["has_security"].value_counts()
count_storage_room = df["has_storage_room"].value_counts()
print(f"Mieszkania, które posiadają windę: {count_elevator}")
print(f"Mieszkania, które posiadają miejsce parkingowe: {count_parking_space}")
print(f"Mieszkania, które posiadają ochronę: {count_security}")
print(f"Mieszkania, które posiadają komórkę lokatorską: {count_storage_room}")