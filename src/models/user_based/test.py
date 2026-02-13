import os
import pickle
import pandas as pd
from metrics import rmse, mae

BASE_DIR = os.path.dirname(__file__)

TEST_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "user_knn.pkl")


def main():

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded")

    # Load test data
    df = pd.read_csv(TEST_PATH)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():

        pred = model.predict(row["user_id"], row["food_id"])

        if pred is not None:
            y_true.append(row["rating"])
            y_pred.append(pred)

    test_rmse = rmse(y_true, y_pred)
    test_mae = mae(y_true, y_pred)

    print("\n🔹 Final Test Results")
    print("RMSE:", test_rmse)
    print("MAE:", test_mae)
    print("NMAE:", test_mae/5)

if __name__ == "__main__":
    main()
