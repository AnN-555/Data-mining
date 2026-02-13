import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedPhoBERT:

    def __init__(self,
                 w_text=0.9,
                 w_nut=0.1,
                 batch_size=16,
                 device=None):

        self.w_text = w_text
        self.w_nut = w_nut
        self.batch_size = batch_size

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.model.to(self.device)
        self.model.eval()

        self.user_mean = None

    # Mean Pooling (better than CLS)
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    # Batch Encode Text
    def encode_text(self, texts):

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):

            batch_texts = texts[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self.mean_pooling(outputs, inputs["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    # Fit model
    def fit(self, df):

        food_df = df.drop_duplicates("food_id").copy()

        self.food_ids = food_df["food_id"].values
        self.food_id_to_index = {
            fid: idx for idx, fid in enumerate(self.food_ids)
        }

        # -------- TEXT --------
        texts = (
            food_df["description"].fillna("") + " " +
            food_df["ingredients"].fillna("") + " " +
            food_df["dish_tags"].fillna("")
        ).tolist()

        print("Encoding text with PhoBERT (batch)...")
        text_embeddings = self.encode_text(texts)

        # Normalize text block
        text_embeddings = normalize(text_embeddings)

        # -------- NUTRITION --------
        nutrient_cols = ["calories", "fat", "fiber", "sugar", "protein"]

        self.scaler = MinMaxScaler()
        nut_scaled = self.scaler.fit_transform(
            food_df[nutrient_cols].fillna(0)
        )

        nut_scaled = normalize(nut_scaled)

        # -------- Combine --------
        self.item_matrix = np.hstack([
            self.w_text * text_embeddings,
            self.w_nut * nut_scaled
        ])

        # Normalize final vector (important)
        self.item_matrix = normalize(self.item_matrix)

        print("PhoBERT content model trained (research version)")

    # Build User Profile
    def build_user_profile(self, user_df):

        ratings = user_df["rating"].values
        self.user_mean = np.mean(ratings)

        profile = np.zeros(self.item_matrix.shape[1])
        total_weight = 0

        for _, row in user_df.iterrows():

            food_id = row["food_id"]
            rating = row["rating"]

            if food_id in self.food_id_to_index:

                idx = self.food_id_to_index[food_id]
                centered = rating - self.user_mean

                profile += centered * self.item_matrix[idx]
                total_weight += abs(centered)

        if total_weight > 0:
            profile /= total_weight

        # Normalize profile
        profile = profile / (np.linalg.norm(profile) + 1e-8)

        return profile

    # Predict rating
    def predict(self, user_profile, food_id):

        if food_id not in self.food_id_to_index:
            return None

        idx = self.food_id_to_index[food_id]

        sim = np.dot(user_profile, self.item_matrix[idx])

        pred = self.user_mean + 2 * sim
        return max(1, min(5, pred))

    # Fast predict all items
    def predict_all(self, user_profile):

        sims = np.dot(self.item_matrix, user_profile)
        preds = self.user_mean + 2 * sims
        preds = np.clip(preds, 1, 5)

        return preds