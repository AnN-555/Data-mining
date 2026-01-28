import pandas as pd
from sklearn.preprocessing import StandardScaler

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


def load_foods(path="data/processed/foods_processed.csv"):
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