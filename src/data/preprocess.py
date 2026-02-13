import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../data/processed")
)

def load_data():
    foods = pd.read_csv(os.path.join(DATA_PATH, "foods_processed.csv"))
    ratings = pd.read_csv(os.path.join(DATA_PATH, "rating_processed.csv"))
    return foods, ratings


def preprocess():
    print("🔹 Loading data...")
    foods, ratings = load_data()

    print("🔹 Encoding IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    ratings["user_idx"] = user_encoder.fit_transform(ratings["userId"])
    ratings["item_idx"] = item_encoder.fit_transform(ratings["foodId"])

    # Giữ foods xuất hiện trong ratings
    foods = foods[foods["food_id"].isin(ratings["foodId"].unique())]

    # Encode food_id trong foods
    foods["item_idx"] = item_encoder.transform(foods["food_id"])

    print("🔹 Creating combined text...")
    text_cols = [
        "dish_name",
        "description",
        "ingredients",
        "cooking_method",
        "dish_tags"
    ]

    for col in text_cols:
        foods[col] = foods[col].fillna("")

    foods["combined_text"] = foods[text_cols].agg(" ".join, axis=1)

    print("🔹 Merging ratings + foods...")
    combined = ratings.merge(
        foods[["item_idx", "combined_text"]],
        on="item_idx",
        how="left"
    )

    print("🔹 Saving combined dataset...")
    combined.to_csv(
        os.path.join(DATA_PATH, "combined_dataset.csv"),
        index=False
    )

    print("✅ Done!")
    print("Users:", combined["user_idx"].nunique())
    print("Items:", combined["item_idx"].nunique())
    print("Total interactions:", len(combined))


if __name__ == "__main__":
    preprocess()
