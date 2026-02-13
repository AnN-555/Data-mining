import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "svd_model.pkl")
TEST_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv"))


def main():
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    user2idx = data["user2idx"]
    item2idx = data["item2idx"]

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    test_data = []
    for _, row in test_df.iterrows():
        if row["user_id"] in user2idx and row["food_id"] in item2idx:
            u = user2idx[row["user_id"]]
            i = item2idx[row["food_id"]]
            r = row["rating"]
            test_data.append((u, i, r))

    preds = model.predict(test_data)
    true = np.array([r for (_, _, r) in test_data])

    rmse = np.sqrt(mean_squared_error(true, preds))
    mae = mean_absolute_error(true, preds)

    print("\nFinal Test Results")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("NMAE:", mae/5)


if __name__ == "__main__":
    main()
