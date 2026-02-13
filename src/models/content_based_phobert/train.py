import os
import json
import pickle
import pandas as pd
from model import ContentBasedPhoBERT


BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/combined_train_val.csv")
)

CONFIG_PATH = os.path.join(BASE_DIR, "best_params.json")
MODEL_PATH = os.path.join(BASE_DIR, "phobert_model.pkl")


def main():

    df = pd.read_csv(DATA_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    w_text, w_nut = config["best_weights"]

    print("Training final model with:", w_text, w_nut)

    model = ContentBasedPhoBERT(
        w_text=w_text,
        w_nut=w_nut
    )

    model.fit(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
