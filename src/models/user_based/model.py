import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k=20, similarity="pearson"):
        """
        similarity: "cosine" hoặc "pearson"
        """
        self.k = k
        self.similarity = similarity
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_means = None

    def fit(self, df):

        # Tạo user-item matrix
        self.user_item_matrix = df.pivot_table(
            index="user_id",
            columns="food_id",
            values="rating"
        )

        # Tính mean rating của từng user
        self.user_means = self.user_item_matrix.mean(axis=1)

        if self.similarity == "pearson":
            self._fit_pearson()

        elif self.similarity == "cosine":
            self._fit_cosine()

        else:
            raise ValueError("similarity must be 'pearson' or 'cosine'")

        print(f"Model trained using {self.similarity} similarity")

    # Pearson Similarity
    def _fit_pearson(self):

        centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
        centered_matrix = centered_matrix.fillna(0)

        self.user_similarity = np.corrcoef(centered_matrix)
        self.user_similarity = np.nan_to_num(self.user_similarity)

    # Cosine Similarity
    def _fit_cosine(self):

        matrix = self.user_item_matrix.fillna(0)

        self.user_similarity = cosine_similarity(matrix)

    # Prediction
    def predict(self, user_id, food_id):

        if user_id not in self.user_item_matrix.index:
            return None

        if food_id not in self.user_item_matrix.columns:
            return None

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        sim_scores = self.user_similarity[user_idx].copy()

        # Loại chính user
        sim_scores[user_idx] = 0

        # Top-k neighbors
        top_k_users = np.argsort(sim_scores)[-self.k:]

        numerator = 0
        denominator = 0

        for idx in top_k_users:

            sim = sim_scores[idx]
            neighbor_rating = self.user_item_matrix.iloc[idx][food_id]

            if not np.isnan(neighbor_rating):

                if self.similarity == "pearson":
                    neighbor_mean = self.user_means.iloc[idx]
                    numerator += sim * (neighbor_rating - neighbor_mean)
                else:  # cosine
                    numerator += sim * neighbor_rating

                denominator += abs(sim)

        if denominator == 0:
            return self.user_means.loc[user_id]

        if self.similarity == "pearson":
            predicted = self.user_means.loc[user_id] + (numerator / denominator)
        else:
            predicted = numerator / denominator

        return predicted