import os
import json
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from model import NCF
from utils import create_mapping, apply_mapping

# CONFIG

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv")
)

BEST_PARAM_PATH = os.path.join(BASE_DIR, "best_params.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "ncf_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN FULL DATA

def train_full(df, params):

    print("DEBUG: creating mapping...")
    user2idx, item2idx = create_mapping(df)

    print("DEBUG: applying mapping...")
    users, items, ratings = apply_mapping(df, user2idx, item2idx)

    dataset = TensorDataset(users, items, ratings)

    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True
    )

    print("DEBUG: total batches =", len(loader))

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

    print("DEBUG: start training...")
    model.train()

    for epoch in range(params["epochs"]):
        total_loss = 0

        for u, i, r in loader:
            u = u.to(DEVICE)
            i = i.to(DEVICE)
            r = r.to(DEVICE)

            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return model, user2idx, item2idx

# MAIN

def main():

    print("DEBUG: loading data...")
    df = pd.read_csv(DATA_PATH)

    print("DEBUG: loading best params...")
    with open(BEST_PARAM_PATH, "r") as f:
        config = json.load(f)

    params = config["best_params"]
    print("Using params:", params)

    model, user2idx, item2idx = train_full(df, params)

    print("DEBUG: saving model...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "params": params
    }, MODEL_SAVE_PATH)

    print("Model saved at:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
