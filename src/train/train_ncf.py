import torch
from torch.utils.data import DataLoader
from src.data.ratings_dataset import RatingsDataset
from src.models.ncf.ncf import NCF

DATA_PATH = "data/processed/ratings_processed.csv"

def main():
    dataset = RatingsDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    num_users = dataset.user_ids.max().item() + 1
    num_items = dataset.food_ids.max().item() + 1

    model = NCF(num_users, num_items)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10):
        total = 0
        for u, i, r in loader:
            pred = model(u, i)
            loss = loss_fn(pred, r)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}: MSE = {total/len(loader):.4f}")

if __name__ == "__main__":
    main()