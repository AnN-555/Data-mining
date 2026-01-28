import numpy as np

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            score += 1 / np.log2(i + 2)

    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return score / ideal if ideal > 0 else 0