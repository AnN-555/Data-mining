import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.evaluation.rating_metrics import mse, rmse, mae
from src.data.ratings_dataset import RatingsDataset

def evaluate_rating(model, test_path):
    ds = RatingsDataset(test_path)
    loader = DataLoader(ds, batch_size=512)

    preds, trues = [], []

    model.eval()
    with torch.no_grad():
        for u, i, r in loader:
            p = model(u, i)
            preds.append(p)
            trues.append(r)

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    return {
        "MSE": mse(preds, trues),
        "RMSE": rmse(preds, trues),
        "MAE": mae(preds, trues)
    }