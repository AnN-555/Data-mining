import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed")
)

def main():
    print("🔹 Loading datasets...")

    ratings = pd.read_csv(os.path.join(DATA_PATH, "rating_processed.csv"))
    combined = pd.read_csv(os.path.join(DATA_PATH, "combined_dataset.csv"))

    # Chuẩn hóa cột
    ratings.columns = ratings.columns.str.strip()
    combined.columns = combined.columns.str.strip()

    ratings = ratings.rename(columns={
        "foodId": "food_id",
        "userId": "user_id"
    })

    combined = combined.rename(columns={
        "foodId": "food_id",
        "userId": "user_id"
    })

    print("Total ratings:", len(ratings))

    # Split rating (90/10)
    rating_train_val, rating_test = train_test_split(
        ratings,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    print("Train+Val:", len(rating_train_val))
    print("Test:", len(rating_test))

    # Split combined theo rating split
    combined_train_val = combined.merge(
        rating_train_val[["user_id", "food_id"]],
        on=["user_id", "food_id"],
        how="inner"
    )

    combined_test = combined.merge(
        rating_test[["user_id", "food_id"]],
        on=["user_id", "food_id"],
        how="inner"
    )

    print("Combined Train+Val:", len(combined_train_val))
    print("Combined Test:", len(combined_test))

    # Save
    rating_train_val.to_csv(
        os.path.join(DATA_PATH, "rating_train_val.csv"),
        index=False
    )

    rating_test.to_csv(
        os.path.join(DATA_PATH, "rating_test.csv"),
        index=False
    )

    combined_train_val.to_csv(
        os.path.join(DATA_PATH, "combined_train_val.csv"),
        index=False
    )

    combined_test.to_csv(
        os.path.join(DATA_PATH, "combined_test.csv"),
        index=False
    )

    print("Split completed & saved!")

if __name__ == "__main__":
    main()
