import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
csv_path_foods_process = BASE_DIR / "data" / "processed" / "foods_processed.csv"
csv_path_raw_rating = BASE_DIR / "data" / "raw" / "raw_rating.csv"


data_of_food = pd.read_csv(csv_path_foods_process)


data_of_rating = pd.read_csv(csv_path_raw_rating)
valid_food_ids = data_of_food["food_id"].unique()

data_of_rating_filtered = data_of_rating[
    data_of_rating["foodId"].isin(valid_food_ids)
]
data_of_rating_filtered.to_csv(BASE_DIR / "data" / "processed" / "rating_processed.csv", index=False)
print(f"Filtered Ratings: {len(data_of_rating_filtered)}")
