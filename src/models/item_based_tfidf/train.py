import os
import json
import pickle
import pandas as pd
from model import ContentBasedTFIDFWeighted

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/combined_train_val.csv")
)

CONFIG_PATH = os.path.join(BASE_DIR, "best_params.json")
MODEL_PATH = os.path.join(BASE_DIR, "content_model.pkl")


def main():

    df = pd.read_csv(DATA_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    best_weights = config["best_weights"]

    model = ContentBasedTFIDFWeighted(
        w_desc=best_weights[0],
        w_ing=best_weights[1],
        w_tag=best_weights[2],
        w_nut=best_weights[3]
    )

    model.fit(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
