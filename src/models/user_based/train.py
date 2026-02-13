import json
import os
import pandas as pd
import pickle
from model import UserBasedCF

BASE_DIR = os.path.dirname(__file__)

TRAIN_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv")
)

MODEL_PATH = os.path.join(BASE_DIR, "user_knn.pkl")

CONFIG_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "best_params.json")
)

def main():

    # Load best params
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    best_k = config["best_k"]
    best_similarity = config["best_similarity"]

    print("Training with:")
    print("k =", best_k)
    print("similarity =", best_similarity)

    # Load data
    df = pd.read_csv(TRAIN_PATH)

    # Train model
    model = UserBasedCF(
        k=best_k,
        similarity=best_similarity
    )

    model.fit(df)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
