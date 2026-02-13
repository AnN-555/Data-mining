import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import NCF

# CONFIG

BASE_DIR = os.path.dirname(__file__)
TEST_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "ncf_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAIN

def main():

    print("DEBUG: loading model...")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    user2idx = checkpoint["user2idx"]
    item2idx = checkpoint["item2idx"]
    params = checkpoint["params"]

    model = NCF(
        num_users=len(user2idx),
        num_items=len(item2idx),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("DEBUG: loading test data...")
    df = pd.read_csv(TEST_PATH)

    preds = []
    targets = []

    print("DEBUG: evaluating...")

    with torch.no_grad():
        for _, row in df.iterrows():

            user = row["user_id"]
            item = row["food_id"]
            rating = row["rating"]

            # Skip unseen users/items
            if user not in user2idx or item not in item2idx:
                continue

            u = torch.tensor([user2idx[user]]).to(DEVICE)
            i = torch.tensor([item2idx[item]]).to(DEVICE)

            pred = model(u, i).item()

            preds.append(pred)
            targets.append(rating)

    # METRICS

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)

    rating_min = min(targets)
    rating_max = max(targets)

    nmae = mae / (rating_max - rating_min)

    print("\nFinal Test Results")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"NMAE : {nmae:.6f}")

if __name__ == "__main__":
    main()