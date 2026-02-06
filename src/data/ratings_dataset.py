import pandas as pd
import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Bỏ các dòng thiếu id/rating nếu có
        df = df.dropna(subset=["userId", "foodId", "rating"])

        # Tạo mapping userId -> index 0..num_users-1
        unique_users = sorted(df["userId"].unique())
        user2idx = {u: i for i, u in enumerate(unique_users)}

        # Tạo mapping foodId -> index 0..num_items-1
        unique_items = sorted(df["foodId"].unique())
        item2idx = {f: i for i, f in enumerate(unique_items)}

        self.user_ids = torch.tensor(
            df["userId"].map(user2idx).values, dtype=torch.long
        )
        self.food_ids = torch.tensor(
            df["foodId"].map(item2idx).values, dtype=torch.long
        )
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        # Lưu lại để dùng khi tạo model
        self.num_users = len(user2idx)
        self.num_items = len(item2idx)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.food_ids[idx],
            self.ratings[idx],
        )