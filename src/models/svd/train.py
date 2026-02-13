import os
import json
import pickle
import pandas as pd

from model import SVD


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv"))
PARAM_PATH = os.path.join(BASE_DIR, "best_params.json")
MODEL_PATH = os.path.join(BASE_DIR, "svd_model.pkl")


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    print("Loading best params...")
    with open(PARAM_PATH) as f:
        best_params = json.load(f)

    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {i: j for j, i in enumerate(df["food_id"].unique())}

    train_data = []
    for _, row in df.iterrows():
        u = user2idx[row["user_id"]]
        i = item2idx[row["food_id"]]
        r = row["rating"]
        train_data.append((u, i, r))

    print("Training final model with:", best_params)

    model = SVD(
        n_users=len(user2idx),
        n_items=len(item2idx),
        **best_params
    )

    model.fit(train_data)

    print("Saving model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "user2idx": user2idx,
            "item2idx": item2idx
        }, f)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
