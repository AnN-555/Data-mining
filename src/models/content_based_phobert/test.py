import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_DIR = os.path.dirname(__file__)

TRAIN_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/combined_train_val.csv")
)

TEST_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "phobert_model.pkl")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nmae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / 5.0


def main():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_true, y_pred = [], []

    for user_id in test_df["user_id"].unique():

        train_user = train_df[train_df["user_id"] == user_id]
        if len(train_user) == 0:
            continue

        user_profile = model.build_user_profile(train_user)
        test_user = test_df[test_df["user_id"] == user_id]

        for _, row in test_user.iterrows():

            pred = model.predict(user_profile, row["food_id"])

            if pred is not None:
                y_true.append(row["rating"])
                y_pred.append(pred)

    print("\nFinal Test Results")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("NMAE:", nmae(y_true, y_pred))


if __name__ == "__main__":
    main()
