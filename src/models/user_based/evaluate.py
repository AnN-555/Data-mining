import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from model import UserBasedCF
from metrics import rmse, mae
import json

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/rating_train_val.csv")
)


def cross_validate_model(k_list, similarities, n_splits=5):

    df = pd.read_csv(DATA_PATH)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    for sim in similarities:
        for k_neighbors in k_list:

            print("\n====================================")
            print(f"Testing k = {k_neighbors} | similarity = {sim}")
            print("====================================")

            rmse_scores = []
            mae_scores = []

            for fold, (train_index, val_index) in enumerate(kf.split(df)):
                print(f"\n🔹 Fold {fold+1}")

                train_df = df.iloc[train_index]
                val_df = df.iloc[val_index]

                model = UserBasedCF(k=k_neighbors, similarity=sim)
                model.fit(train_df)

                y_true = []
                y_pred = []

                for _, row in val_df.iterrows():
                    pred = model.predict(row["user_id"], row["food_id"])
                    if pred is not None:
                        y_true.append(row["rating"])
                        y_pred.append(pred)

                fold_rmse = rmse(y_true, y_pred)
                fold_mae = mae(y_true, y_pred)

                rmse_scores.append(fold_rmse)
                mae_scores.append(fold_mae)

                print("RMSE:", fold_rmse)
                print("MAE:", fold_mae)

            mean_rmse = np.mean(rmse_scores)
            mean_mae = np.mean(mae_scores)

            results[(k_neighbors, sim)] = mean_rmse

            print("\nCV Results")
            print("Mean RMSE:", mean_rmse)
            print("Mean MAE:", mean_mae)

    # Chọn best combination
    best_params = min(results, key=results.get)
    best_k, best_similarity = best_params
    best_rmse = results[best_params]

    print("\n===================================")
    print("BEST PARAMETERS FOUND")
    print("k:", best_k)
    print("similarity:", best_similarity)
    print("BEST RMSE:", best_rmse)
    print("===================================")

    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "best_params.json"))

    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {
                "best_k": best_k,
                "best_similarity": best_similarity,
                "best_rmse": best_rmse,
            },
            f,
        )

    print(f"Best params saved to {CONFIG_PATH}")

    return best_params, results


if __name__ == "__main__":
    k_values = [5, 10, 20, 30, 40, 50]
    similarities = ["cosine", "pearson"]

    cross_validate_model(k_values, similarities)
