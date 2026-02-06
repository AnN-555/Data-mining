import pandas as pd
import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.user_ids = torch.tensor(df["userId"].values, dtype=torch.long)
        # Convert food IDs to zero-based indices to align with embeddings
        self.food_ids = torch.tensor(df["foodId"].values - 1, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.food_ids[idx],
            self.ratings[idx],
        )