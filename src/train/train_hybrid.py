import torch
import pandas as pd
from src.models.hybrid.hybrid import HybridModel
from src.models.content.phobert import PhoBERTEncoder
from src.data.ratings_dataset import RatingsDataset
from torch.utils.data import DataLoader

FOOD_PATH = "data/processed/foods_processed.csv"
RATING_PATH = "data/processed/ratings_processed.csv"

def main():
    food_df = pd.read_csv(FOOD_PATH)
    rating_ds = RatingsDataset(RATING_PATH)
    loader = DataLoader(rating_ds, batch_size=64, shuffle=True)

    encoder = PhoBERTEncoder()
    text_emb = encoder.encode(food_df["description"].fillna("").tolist())

    nutri = torch.tensor(
        food_df[["calories","fat","fiber","sugar","protein"]].values,
        dtype=torch.float32
    )

    model = HybridModel(
        num_users=rating_ds.user_ids.max()+1,
        num_items=rating_ds.food_ids.max()+1
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        total = 0
        for u, i, r in loader:
            pred = model(u, i, text_emb[i], nutri[i])
            loss = loss_fn(pred, r)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}: Hybrid MSE = {total/len(loader):.4f}")

if __name__ == "__main__":
    main()