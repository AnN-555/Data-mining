import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from model import NCF

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv")
)

SAVE_PATH = os.path.join(BASE_DIR, "best_params.json")


def create_mapping(df):
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {i: j for j, i in enumerate(df["food_id"].unique())}
    return user2idx, item2idx


def convert(df, user2idx, item2idx):
    users = df["user_id"].map(user2idx).values
    items = df["food_id"].map(item2idx).values
    ratings = df["rating"].values
    return users, items, ratings


def train_one_fold(train_df, val_df, params):
    user2idx, item2idx = create_mapping(train_df)

    u_train, i_train, r_train = convert(train_df, user2idx, item2idx)
    u_val, i_val, r_val = convert(val_df, user2idx, item2idx)

    model = NCF(
        num_users=len(user2idx),
        num_items=len(item2idx),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.MSELoss()

    model.train()

    for _ in range(params["epochs"]):
        for u, i, r in zip(u_train, i_train, r_train):
            u = torch.tensor([u])
            i = torch.tensor([i])
            r = torch.tensor([r], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()

    # validation
    model.eval()
    preds = []
    with torch.no_grad():
        for u, i in zip(u_val, i_val):
            u = torch.tensor([u])
            i = torch.tensor([i])
            pred = model(u, i).item()
            preds.append(pred)

    rmse = np.sqrt(mean_squared_error(r_val, preds))
    return rmse


def main():
    df = pd.read_csv(DATA_PATH)

    param_grid = [
        {"embedding_dim": 32, "hidden_dim": 64, "lr": 0.001, "epochs": 5},
        {"embedding_dim": 64, "hidden_dim": 128, "lr": 0.001, "epochs": 5},
    ]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    best_rmse = float("inf")
    best_params = None

    for params in param_grid:
        print("Testing:", params)
        rmses = []

        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            rmse = train_one_fold(train_df, val_df, params)
            rmses.append(rmse)

        avg_rmse = np.mean(rmses)
        print("Avg RMSE:", avg_rmse)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    with open(SAVE_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best params:", best_params)


if __name__ == "__main__":
    main()
