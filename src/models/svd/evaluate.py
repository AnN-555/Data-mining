import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from model import SVD

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv"))
SAVE_PATH = os.path.join(BASE_DIR, "best_params.json")

def create_mapping(df):
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {i: j for j, i in enumerate(df["food_id"].unique())}
    return user2idx, item2idx


def convert(df, user2idx, item2idx):
    data = []
    for _, row in df.iterrows():
        u = user2idx[row["user_id"]]
        i = item2idx[row["food_id"]]
        r = row["rating"]
        data.append((u, i, r))
    return data


def run_cv(params):
    df = pd.read_csv(DATA_PATH)
    user2idx, item2idx = create_mapping(df)
    data = convert(df, user2idx, item2idx)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, val_idx in kf.split(data):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        model = SVD(
            n_users=len(user2idx),
            n_items=len(item2idx),
            **params
        )

        model.fit(train_data)

        preds = model.predict(val_data)
        true = np.array([r for (_, _, r) in val_data])

        rmse = np.sqrt(mean_squared_error(true, preds))
        rmses.append(rmse)

    return np.mean(rmses)


def main():
    param_grid = [
        {"k": 64,  "lr": 0.003, "reg": 0.02,  "epochs": 40},
        {"k": 128, "lr": 0.003, "reg": 0.02,  "epochs": 50},
        {"k": 128, "lr": 0.002, "reg": 0.015, "epochs": 60},
        {"k": 256, "lr": 0.002, "reg": 0.02,  "epochs": 60},
    ]

    best_rmse = float("inf")
    best_params = None

    for params in param_grid:
        print("Testing:", params)
        rmse = run_cv(params)
        print("CV RMSE:", rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("Best:", best_params, "RMSE:", best_rmse)

    with open(SAVE_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    print("Saved best_params.json")

if __name__ == "__main__":
    main()