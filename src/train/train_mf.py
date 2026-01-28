import torch
from torch.utils.data import DataLoader
from src.data.ratings_dataset import RatingsDataset
from src.models.mf.mf import MatrixFactorization

DATA_PATH = "data/processed/ratings_processed.csv"
EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3

def main():
    dataset = RatingsDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_users = dataset.user_ids.max().item() + 1
    num_items = dataset.food_ids.max().item() + 1

    model = MatrixFactorization(num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for user, item, rating in loader:
            pred = model(user, item)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: MSE = {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    main()