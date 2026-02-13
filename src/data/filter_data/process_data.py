import pandas as pd
import numpy as np

from pathlib import Path

# Determine the base directory
BASE_DIR = Path(__file__).resolve().parents[3]
csv_path = BASE_DIR / "data" / "raw" / "raw_foods_data.csv"

# check null values 
def checkNullValues(df):
    list_data_of_food_np_filter = df.dropna()
    return list_data_of_food_np_filter

# Read and process the data
list_data_of_food = pd.read_csv(csv_path)
# drop null values
list_data_of_food = checkNullValues(list_data_of_food)
# select specific columns
list_data_of_food_np_filter = list_data_of_food.loc[:, ["food_id","dish_name", "description", "ingredients", "cooking_method", "dish_tags", "calories", "fat", "fiber", "sugar", "protein"]]

calories = list_data_of_food_np_filter["sugar"]*4 + list_data_of_food_np_filter["protein"]*4 + list_data_of_food_np_filter["fat"]*9

filtered_foods = list_data_of_food_np_filter[
    (list_data_of_food_np_filter["sugar"] <=17) 
    & (list_data_of_food_np_filter["fiber"] >= 5)
] 
print(f"Filtered Foods: {filtered_foods.sample(20)}")
print(len(filtered_foods))

filtered_foods.to_csv(BASE_DIR / "data" / "processed" / "foods_processed.csv", index=False)