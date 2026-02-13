import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


class ContentBasedTFIDFWeighted:

    def __init__(self,
                 w_desc=0.25,
                 w_ing=0.6,
                 w_tag=0.1,
                 w_nut=0.05):

        self.w_desc = w_desc
        self.w_ing = w_ing
        self.w_tag = w_tag
        self.w_nut = w_nut

        # user statistics (set when building profile)
        self.user_mean = None

    # TRAIN ITEM FEATURES
    def fit(self, df):

        # Unique food items
        food_df = df.drop_duplicates("food_id").copy()

        self.food_ids = food_df["food_id"].values
        self.food_id_to_index = {
            food_id: idx for idx, food_id in enumerate(self.food_ids)
        }

        # ================= TEXT FEATURES =================
        self.desc_vectorizer = TfidfVectorizer()
        self.ing_vectorizer = TfidfVectorizer()
        self.tag_vectorizer = TfidfVectorizer()

        desc_vec = self.desc_vectorizer.fit_transform(
            food_df["description"].fillna("")
        )

        ing_vec = self.ing_vectorizer.fit_transform(
            food_df["ingredients"].fillna("")
        )

        tag_vec = self.tag_vectorizer.fit_transform(
            food_df["dish_tags"].fillna("")
        )

        # ================= NUMERIC FEATURES =================
        nutrient_cols = ["calories", "fat", "fiber", "sugar", "protein"]

        self.scaler = MinMaxScaler()
        nut_scaled = self.scaler.fit_transform(
            food_df[nutrient_cols].fillna(0)
        )

        nut_vec = csr_matrix(nut_scaled)

        # ================= CONCAT WEIGHTED FEATURES =================
        self.item_matrix = hstack([
            self.w_desc * desc_vec,
            self.w_ing * ing_vec,
            self.w_tag * tag_vec,
            self.w_nut * nut_vec
        ]).tocsr()

        print("Content model trained")

    # BUILD USER PROFILE (CENTERED RATING)
    def build_user_profile(self, user_df):

        profile = np.zeros(self.item_matrix.shape[1])

        ratings = user_df["rating"].values
        self.user_mean = np.mean(ratings)

        total_weight = 0

        for _, row in user_df.iterrows():

            food_id = row["food_id"]
            rating = row["rating"]

            if food_id in self.food_id_to_index:

                idx = self.food_id_to_index[food_id]
                item_vec = self.item_matrix[idx].toarray().flatten()

                # Centered rating (very important)
                centered_rating = rating - self.user_mean

                profile += centered_rating * item_vec
                total_weight += abs(centered_rating)

        if total_weight > 0:
            profile /= total_weight

        return profile

    # PREDICT RATING
    def predict(self, user_profile, food_id):

        if food_id not in self.food_id_to_index:
            return None

        idx = self.food_id_to_index[food_id]

        item_vector = self.item_matrix[idx].toarray()
        user_profile = user_profile.reshape(1, -1)

        sim = cosine_similarity(user_profile, item_vector)[0][0]

        # Add back user mean
        predicted_rating = self.user_mean + 2 * sim

        # Clamp to rating scale
        predicted_rating = max(1, min(5, predicted_rating))

        return predicted_rating
