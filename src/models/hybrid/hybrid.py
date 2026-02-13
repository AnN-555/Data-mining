import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()

        embed_dim = 32

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        # Project PhoBERT 768 → 64
        self.text_fc = nn.Linear(768, 64)

        # Project nutrition 5 → 16
        self.nutri_fc = nn.Linear(5, 16)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2 + 64 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_id, item_id, text_feat, nutri_feat):
        u = self.user_emb(user_id)
        i = self.item_emb(item_id)

        t = self.text_fc(text_feat)
        n = self.nutri_fc(nutri_feat)

        x = torch.cat([u, i, t, n], dim=1)

        out = self.fc(x)

        return out.squeeze()
