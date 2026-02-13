import torch
import pandas as pd
from src.models.hybrid.hybrid import HybridModel
from src.models.content.phobert import PhoBERTEncoder
from src.data.ratings_dataset import RatingsDataset
from torch.utils.data import DataLoader
from pathlib import Path

# ========================
# Load Paths
# ========================
BASE_DIR = Path(__file__).resolve().parents[2]
FOOD_PATH = BASE_DIR / "data" / "processed" / "foods_processed.csv"
RATING_PATH = BASE_DIR / "data" / "processed" / "rating_processed.csv"

# ========================
# Load Food Data
# ========================
food_df = pd.read_csv(FOOD_PATH)

food_df = food_df.reset_index(drop=True)
food_df["food_index"] = food_df.index

foodid_to_index = dict(
    zip(food_df["food_id"], food_df["food_index"])
)

num_items = len(food_df)

# ========================
# Load Rating Data
# ========================
rating_df = pd.read_csv(RATING_PATH)

rating_df["food_index"] = rating_df["foodId"].map(foodid_to_index)
rating_df = rating_df.dropna()
rating_df["food_index"] = rating_df["food_index"].astype(int)

num_users = int(rating_df["userId"].max()) + 1

# ========================
# Nutrition tensor
# ========================
nutrition = torch.tensor(
    food_df[["calories","fat","fiber","sugar","protein"]].values,
    dtype=torch.float32
)

# ========================
# Encode PhoBERT (chỉ 1 lần)
# ========================
encoder = PhoBERTEncoder()

text_emb = encoder.encode(
    food_df["dish_name"].fillna("").tolist()
)

print("Text embedding shape:", text_emb.shape)

# ========================
# Dataset + Loader
# ========================
dataset = RatingsDataset(rating_df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========================
# Model
# ========================
model = HybridModel(num_users, num_items)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# ========================
# Train
# ========================
for epoch in range(5):
    total_loss = 0

    for user, item, rating in loader:
        nutri_batch = nutrition[item]
        text_batch = text_emb[item]

        pred = model(user, item, text_batch, nutri_batch)
        loss = loss_fn(pred, rating)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# ========================
# Save model
# ========================
torch.save(model.state_dict(), "hybrid_model.pt")
print("Model saved!")


model.eval()

with torch.no_grad():
    user = torch.tensor([0])
    item = torch.tensor([0])

    pred = model(
        user,
        item,
        text_emb[item],
        nutrition[item]
    )

    print(pred)