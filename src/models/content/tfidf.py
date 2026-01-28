import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )

    def fit(self, texts):
        self.tfidf = self.vectorizer.fit_transform(texts)

    def similarity(self):
        return cosine_similarity(self.tfidf)