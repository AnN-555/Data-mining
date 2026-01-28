import pandas as pd
import numpy as np
from collections import defaultdict
from src.evaluation.topk_metrics import *

def evaluate_topk(model, ratings_df, num_items, k=10):
    user_items = defaultdict(list)

    for _, row in ratings_df.iterrows():
        user_items[row.user_id].append(row.food_id)

    precisions, recalls, ndcgs = [], [], []

    for user, relevant in user_items.items():
        scores = []

        for item in range(num_items):
            score = model.predict(user, item)
            scores.append((item, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [i for i, _ in scores]

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    return {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs)
    }