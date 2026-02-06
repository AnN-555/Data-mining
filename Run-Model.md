python -m src.train.train_mf

python -m src.train.train_tfidf

python -m src.train.train_ncf

python -m src.train.train_phobert

python -m src.train.train_hybrid

# Khuyến nghị theo chế độ tiểu đường (Content-based + diabetic filter)
python -m src.recommend_diabetic --food_id 0 --top_k 15 --diabetic