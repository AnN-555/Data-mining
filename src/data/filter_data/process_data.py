import pandas as pd
import numpy as np

from pathlib import Path

# Determine the base directory
BASE_DIR = Path(__file__).resolve().parents[3]
csv_path = BASE_DIR / "data" / "processed" / "foods_processed.csv"

# check null values 
def check_null_values(df):
    list_data_of_food_np_filter = df.dropna()
    return list_data_of_food_np_filter


# Read and process the data
list_data_of_food = pd.read_csv(csv_path)
# drop null values
list_data_of_food = check_null_values(list_data_of_food)
# select specific columns
list_data_of_food_np_filter = list_data_of_food.loc[:, ["dish_name", "calories", "fat", "fiber", "sugar", "protein"]]


filtered_foods = list_data_of_food_np_filter[
    (list_data_of_food_np_filter["calories"] <= 600) &
    (list_data_of_food_np_filter["sugar"] <= 10) &
    (list_data_of_food_np_filter["fiber"] >= 5) &
    (list_data_of_food_np_filter["fat"] <= 20) &
    (list_data_of_food_np_filter["protein"].between(15, 30))
]
print(f"Filtered Foods:{filtered_foods}")
