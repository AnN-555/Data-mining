import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parents[2]
csv_path = BASE_DIR / "data" / "processed" / "rating_processed.csv"


class RatingsDataset(Dataset):
    def __init__(self, rating_df):
        self.user_ids = torch.tensor(
            rating_df["userId"].values,
            dtype=torch.long
        )
        self.food_ids = torch.tensor(
            rating_df["food_index"].values,
            dtype=torch.long
        )
        self.ratings = torch.tensor(
            rating_df["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.food_ids[idx],
            self.ratings[idx]
        )
