"""
Ràng buộc chế độ ăn cho người tiểu đường (Health-aware filtering).
Dùng để lọc hoặc re-rank danh sách món sau khi recommend.
"""
import pandas as pd
import numpy as np
from typing import Optional

# Ngưỡng mặc định theo khuyến nghị chế độ tiểu đường (có thể điều chỉnh)
DEFAULT_MAX_SUGAR_PER_SERVING_G = 20.0   # gram đường / phần ăn
DEFAULT_MIN_FIBER_G = 2.0                # gram chất xơ tối thiểu
DEFAULT_MAX_CALORIES_PER_SERVING = 500   # kcal / phần (tùy chọn, None = không giới hạn)


def filter_diabetic_safe(
    food_df: pd.DataFrame,
    food_ids: list,
    max_sugar_g: float = DEFAULT_MAX_SUGAR_PER_SERVING_G,
    min_fiber_g: float = DEFAULT_MIN_FIBER_G,
    max_calories: Optional[float] = DEFAULT_MAX_CALORIES_PER_SERVING,
    sugar_col: str = "sugar",
    fiber_col: str = "fiber",
    calories_col: str = "calories",
) -> pd.DataFrame:
    """
    Lọc chỉ giữ các món thỏa ràng buộc tiểu đường.
    food_df: DataFrame có cột food_id (hoặc index trùng food_id) và sugar, fiber, calories.
    food_ids: danh sách food_id cần xét (thường là top-K từ recommender).
    Trả về: subset của food_df tương ứng food_ids và thỏa điều kiện.
    """
    if "food_id" in food_df.columns:
        sub = food_df[food_df["food_id"].isin(food_ids)].copy()
    else:
        sub = food_df.loc[food_df.index.isin(food_ids)].copy() if food_df.index.name else food_df.loc[food_ids].copy()

    mask = (
        (sub[sugar_col] <= max_sugar_g) &
        (sub[fiber_col] >= min_fiber_g)
    )
    if max_calories is not None and calories_col in sub.columns:
        mask = mask & (sub[calories_col] <= max_calories)

    return sub[mask]


def diabetic_score(
    food_df: pd.DataFrame,
    sugar_col: str = "sugar",
    fiber_col: str = "fiber",
    max_sugar: float = 30.0,
    max_fiber: float = 15.0,
) -> np.ndarray:
    """
    Điểm “diabetic-friendly” cho từng món: càng thấp đường, cao xơ thì điểm càng cao.
    Chuẩn hóa theo max_sugar, max_fiber để score nằm trong [0, 1] (1 = tốt nhất).
    """
    sugar = np.clip(food_df[sugar_col].fillna(0).values, 0, max_sugar)
    fiber = np.clip(food_df[fiber_col].fillna(0).values, 0, max_fiber)
    # score: ưu tiên ít đường, nhiều xơ
    sugar_norm = 1.0 - (sugar / max_sugar)
    fiber_norm = fiber / max_fiber
    return 0.5 * sugar_norm + 0.5 * fiber_norm


def filter_and_rerank_by_diabetic(
    food_df: pd.DataFrame,
    food_ids: list,
    scores: Optional[list] = None,
    max_sugar_g: float = DEFAULT_MAX_SUGAR_PER_SERVING_G,
    min_fiber_g: float = DEFAULT_MIN_FIBER_G,
    max_calories: Optional[float] = DEFAULT_MAX_CALORIES_PER_SERVING,
    use_diabetic_score: bool = True,
) -> list:
    """
    Lọc món theo ràng buộc tiểu đường, rồi (tùy chọn) re-rank theo điểm diabetic.
    - food_df: DataFrame có cột food_id, sugar, fiber, calories.
    - food_ids: danh sách food_id (thứ tự từ recommender).
    - scores: điểm recommend gốc (cùng thứ tự food_ids); nếu None thì chỉ dùng diabetic_score.
    Trả về: danh sách food_id sau khi lọc và sắp xếp (ưu tiên diabetic-safe rồi điểm).
    """
    if not food_ids:
        return []

    has_food_id = "food_id" in food_df.columns
    if has_food_id:
        sub = food_df[food_df["food_id"].isin(food_ids)].copy()
    else:
        sub = food_df.loc[food_df.index.intersection(food_ids)].copy()
        sub = sub.loc[sub.index.isin(food_ids)]

    if sub.empty:
        return []

    # Lọc theo ngưỡng
    mask = (
        (sub["sugar"].fillna(0) <= max_sugar_g) &
        (sub["fiber"].fillna(0) >= min_fiber_g)
    )
    if max_calories is not None and "calories" in sub.columns:
        mask = mask & (sub["calories"].fillna(np.inf) <= max_calories)
    sub = sub[mask].copy()

    if sub.empty:
        return []

    ids_in_sub = sub["food_id"].tolist() if has_food_id else sub.index.tolist()

    d_score = diabetic_score(sub)
    if use_diabetic_score:
        if scores is not None and len(scores) == len(food_ids):
            id_to_score = dict(zip(food_ids, scores))
            rec_scores = np.array([id_to_score.get(fid, 0.0) for fid in ids_in_sub])
            rmax = np.max(rec_scores)
            rec_norm = rec_scores / (rmax + 1e-9) if rmax > 0 else np.ones_like(rec_scores)
            combined = 0.6 * rec_norm + 0.4 * d_score
        else:
            combined = d_score
    else:
        combined = np.array([dict(zip(food_ids, scores)).get(fid, 0.0) for fid in ids_in_sub]) if scores else d_score

    sub["_combined_score"] = combined
    sub = sub.sort_values("_combined_score", ascending=False)
    return sub["food_id"].tolist() if has_food_id else sub.index.tolist()


def diabetic_compliance_rate(
    food_df: pd.DataFrame,
    recommended_food_ids: list,
    max_sugar_g: float = DEFAULT_MAX_SUGAR_PER_SERVING_G,
    min_fiber_g: float = DEFAULT_MIN_FIBER_G,
) -> float:
    """
    Tỷ lệ món trong danh sách recommend thỏa ràng buộc tiểu đường (để báo cáo trong bài báo).
    """
    if not recommended_food_ids:
        return 0.0
    sub = food_df[food_df["food_id"].isin(recommended_food_ids)]
    if sub.empty:
        return 0.0
    ok = (sub["sugar"] <= max_sugar_g) & (sub["fiber"] >= min_fiber_g)
    return ok.mean()
