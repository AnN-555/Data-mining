import pandas as pd
from src.models.content.tfidf import TFIDFRecommender

DATA_PATH = "data/processed/foods_processed.csv"

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