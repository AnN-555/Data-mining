import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed")
)

def main():
    print("🔹 Loading datasets...")

    ratings = pd.read_csv(os.path.join(DATA_PATH, "rating_processed.csv"))
    foods = pd.read_csv(os.path.join(DATA_PATH, "foods_processed.csv"))

    print("Ratings samples:", len(ratings))
    print("Foods samples:", len(foods))

    # Chuẩn hóa tên cột
    ratings.columns = ratings.columns.str.strip()
    foods.columns = foods.columns.str.strip()

    ratings = ratings.rename(columns={
        "foodId": "food_id",
        "userId": "user_id"
    })

    # Nếu foods có id khác, đổi về food_id
    foods = foods.rename(columns={
        "foodId": "food_id"
    })

    # Merge
    combined = ratings.merge(
        foods,
        on="food_id",
        how="inner"   # chỉ giữ những food có trong foods
    )

    print("Combined samples:", len(combined))

    # Save
    combined.to_csv(
        os.path.join(DATA_PATH, "combined_dataset.csv"),
        index=False
    )

    print("Saved: combined_dataset.csv")

if __name__ == "__main__":
    main()