import pandas as pd
from src.models.content.phobert import PhoBERTEncoder

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "foods_processed.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    texts = df["description"].fillna("").tolist()

    encoder = PhoBERTEncoder()
    emb = encoder.encode(texts[:16])  # demo batch nhỏ

    print("PhoBERT embedding shape:", emb.shape)

if __name__ == "__main__":
    main()