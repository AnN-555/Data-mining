import pandas as pd
from src.models.content.tfidf import TFIDFRecommender

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "foods_processed.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    text = (
        df["dish_name"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["ingredients"].fillna("") + " " +
        df["dish_tags"].fillna("")
    )

    model = TFIDFRecommender()
    model.fit(text)

    sim = model.similarity()
    print("TF-IDF similarity matrix shape:", sim.shape)

if __name__ == "__main__":
    main()