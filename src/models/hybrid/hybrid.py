import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, num_users, num_items, text_dim=768, nutri_dim=5):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, 32)
        self.item_emb = nn.Embedding(num_items, 32)

        self.fc = nn.Sequential(
            nn.Linear(32+32+text_dim+nutri_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, u, i, text_emb, nutri):
        x = torch.cat([
            self.user_emb(u),
            self.item_emb(i),
            text_emb,
            nutri
        ], dim=1)
        return self.fc(x).squeeze()