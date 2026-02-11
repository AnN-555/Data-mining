import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parents[2]
csv_path = BASE_DIR / "data" / "processed" / "rating_processed.csv"


class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.user_ids = torch.tensor(df["userId"].values, dtype=torch.long)
        self.food_ids = torch.tensor(df["foodId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.food_ids[idx],
            self.ratings[idx],
        )
    

if __name__ == "__main__":
    dataset = RatingsDataset(csv_path)

    print("Dataset length:", len(dataset))

    # test 1 sample
    user_id, food_id, rating = dataset[0]

    print("Sample 0:")
    print("User ID:", user_id)
    print("Food ID:", food_id)
    print("Rating:", rating)

    # test shape tensor
    print("User tensor shape:", dataset.user_ids.shape)
    print("Food tensor shape:", dataset.food_ids.shape)
    print("Rating tensor shape:", dataset.ratings.shape)
