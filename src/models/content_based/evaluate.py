import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from model import ContentBasedTFIDFWeighted


BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../data/processed/combined_train_val.csv")
)

CONFIG_PATH = os.path.join(BASE_DIR, "best_params.json")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_weights(weights, df, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):

        print(f"\nFold {fold+1}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        model = ContentBasedTFIDFWeighted(
            w_desc=weights[0],
            w_ing=weights[1],
            w_tag=weights[2],
            w_nut=weights[3]
        )

        model.fit(train_df)

        y_true, y_pred = [], []

        for user_id in val_df["user_id"].unique():

            train_user_data = train_df[train_df["user_id"] == user_id]

            if len(train_user_data) == 0:
                continue

            user_profile = model.build_user_profile(train_user_data)
            val_user_data = val_df[val_df["user_id"] == user_id]

            for _, row in val_user_data.iterrows():

                pred = model.predict(user_profile, row["food_id"])

                if pred is not None:
                    y_true.append(row["rating"])
                    y_pred.append(pred)

        fold_rmse = rmse(y_true, y_pred)
        rmse_scores.append(fold_rmse)

        print("Fold RMSE:", fold_rmse)

    return np.mean(rmse_scores)


def main():

    df = pd.read_csv(DATA_PATH)

    weight_candidates = [
        (0.3, 0.5, 0.1, 0.1),
        (0.2, 0.6, 0.1, 0.1),
        (0.1, 0.7, 0.1, 0.1),
        (0.25, 0.6, 0.1, 0.05)
    ]

    best_rmse = float("inf")
    best_weights = None

    for weights in weight_candidates:

        print("\nTesting weights:", weights)

        score = evaluate_weights(weights, df)

        print("Average RMSE:", score)

        if score < best_rmse:
            best_rmse = score
            best_weights = weights

    print("\nBest weights:", best_weights)
    print("Best RMSE:", best_rmse)

    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "best_weights": best_weights,
            "best_rmse": best_rmse
        }, f)

    print("Best params saved to:", CONFIG_PATH)


if __name__ == "__main__":
    main()
