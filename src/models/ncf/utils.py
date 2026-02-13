import torch
import pandas as pd


def create_mapping(df):
    user_ids = df["user_id"].unique()
    item_ids = df["food_id"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}

    print("DEBUG: num_users =", len(user2idx))
    print("DEBUG: num_items =", len(item2idx))

    return user2idx, item2idx


def apply_mapping(df, user2idx, item2idx):
    if not set(df["user_id"]).issubset(user2idx.keys()):
        raise ValueError("Found unknown user_id in validation/test set")

    if not set(df["food_id"]).issubset(item2idx.keys()):
        raise ValueError("Found unknown food_id in validation/test set")

    users = df["user_id"].map(user2idx).values
    items = df["food_id"].map(item2idx).values
    ratings = df["rating"].values

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(ratings, dtype=torch.float32),
    )
