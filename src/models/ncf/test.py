import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from model import NCF

BASE_DIR = os.path.dirname(__file__)

TEST_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_test.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "ncf_model.pt")
MAPPING_PATH = os.path.join(BASE_DIR, "mapping.pkl")
PARAM_PATH = os.path.join(BASE_DIR, "best_params.json")


def main():
    df = pd.read_csv(TEST_PATH)

    with open(MAPPING_PATH, "rb") as f:
        user2idx, item2idx = pickle.load(f)

    with open(PARAM_PATH) as f:
        params = json.load(f)

    model = NCF(
        num_users=len(user2idx),
        num_items=len(item2idx),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"]
    )

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    preds = []
    true = []

    with torch.no_grad():
        for _, row in df.iterrows():
            if row["user_id"] not in user2idx:
                continue
            if row["food_id"] not in item2idx:
                continue

            u = torch.tensor([user2idx[row["user_id"]]])
            i = torch.tensor([item2idx[row["food_id"]]])

            pred = model(u, i).item()

            preds.append(pred)
            true.append(row["rating"])

    rmse = np.sqrt(mean_squared_error(true, preds))
    print("Test RMSE:", rmse)


if __name__ == "__main__":
    main()
