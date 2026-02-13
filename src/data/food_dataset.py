import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
TEXT_COLS = [
    "dish_name",
    "description",
    "ingredients",
    "cooking_method",
    "dish_tags"
]

NUTRITION_COLS = [
    "calories",
    "fat",
    "fiber",
    "sugar",
    "protein"
]

BASE_DIR = Path(__file__).resolve().parents[2]
csv_path = BASE_DIR / "data" / "processed" / "foods_processed.csv"


def load_foods(path=csv_path):
    """
    Load processed food dataset
    """
    df = pd.read_csv(path)

    # đảm bảo food_id là int
    df["food_id"] = df["food_id"].astype(int)

    return df


def build_text_feature(df):
    """
    Gộp text cho TF-IDF / PhoBERT
    """
    df["text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
    return df


def build_nutrition_feature(df, scale=True):
    """
    Chuẩn hóa nutrition vector
    """
    X = df[NUTRITION_COLS].values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X

if __name__ == "__main__":
    df = load_foods()
    print("Load OK:", df.shape)

    df = build_text_feature(df)
    print("Text feature OK")

    X = build_nutrition_feature(df)
    print("Nutrition feature shape:", X.shape)

    print("Done 🚀")
