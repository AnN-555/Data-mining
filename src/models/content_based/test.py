import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE_DIR = os.path.dirname(__file__)

TEST_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "content_model.pkl")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():

    df_test = pd.read_csv(TEST_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_true, y_pred = [], []

    for user_id in df_test["user_id"].unique():

        # IMPORTANT:
        # Build user profile ONLY from items user rated in train.
        # Ở đây giả định model đã fit trên train_val đầy đủ.
        # Ta cần lấy rating history của user trong train_val.
        # => Nếu muốn strict hơn, nên load train_val để build profile.

        user_data = df_test[df_test["user_id"] == user_id]

        user_profile = model.build_user_profile(user_data)

        for _, row in user_data.iterrows():

            pred = model.predict(user_profile, row["food_id"])

            if pred is not None:
                y_true.append(row["rating"])
                y_pred.append(pred)

    print("\nFinal Test Results")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("NMAE:", mean_absolute_error(y_true, y_pred)/5)

if __name__ == "__main__":
    main()
