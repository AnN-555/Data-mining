import os
import json
import torch
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from model import NCF
from utils import create_mapping, apply_mapping

# CONFIG

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv")
)

SAVE_PATH = os.path.join(BASE_DIR, "best_params.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN 1 FOLD

def train_one_fold(train_df, val_df, params):

    user2idx, item2idx = create_mapping(train_df)

    u_train, i_train, r_train = apply_mapping(train_df, user2idx, item2idx)
    u_val, i_val, r_val = apply_mapping(val_df, user2idx, item2idx)

    train_dataset = TensorDataset(u_train, i_train, r_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    model = NCF(
        num_users=len(user2idx),
        num_items=len(item2idx),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    criterion = torch.nn.MSELoss()

    # ---- Training ----
    model.train()
    for epoch in range(params["epochs"]):
        total_loss = 0

        for users, items, ratings in train_loader:
            users = users.to(DEVICE)
            items = items.to(DEVICE)
            ratings = ratings.to(DEVICE)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"   Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # ---- Validation ----
    model.eval()
    preds = []

    with torch.no_grad():
        for u, i in zip(u_val, i_val):
            u = u.unsqueeze(0).to(DEVICE)
            i = i.unsqueeze(0).to(DEVICE)
            pred = model(u, i).item()
            preds.append(pred)

    rmse = np.sqrt(mean_squared_error(r_val.numpy(), preds))

    return rmse

# MAIN GRID SEARCH 5CV
def main():

    df = pd.read_csv(DATA_PATH)
    print("DEBUG: total rows =", len(df))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Grid Search Space ----
    param_grid = {
        "embedding_dim": [16, 32],
        "hidden_dim": [64, 128],
        "lr": [0.001, 0.0005],
        "batch_size": [512, 1024],
        "epochs": [5],
        "weight_decay": [0.0, 1e-5]
    }

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    best_rmse = float("inf")
    best_params = None

    print("\nTOTAL CONFIGS:", len(combinations))
    print("=" * 50)

    # ---- Loop all configs ----
    for idx, values in enumerate(combinations):

        params = dict(zip(keys, values))

        print(f"\nCONFIG {idx+1}/{len(combinations)}")
        print("Params:", params)

        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):

            print(f"\n   Fold {fold+1}")

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            rmse = train_one_fold(train_df, val_df, params)
            print(f"   Fold RMSE: {rmse:.4f}")

            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)

        print("\n>>> Mean CV RMSE:", mean_rmse)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params
            print("NEW BEST!")

    # ---- Save Best ----
    print("\n" + "=" * 50)
    print("BEST PARAMS:", best_params)
    print("BEST RMSE:", best_rmse)

    with open(SAVE_PATH, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_rmse": best_rmse
        }, f, indent=4)


if __name__ == "__main__":
    main()
