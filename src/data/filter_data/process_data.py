import pandas as pd
import numpy as np

from pathlib import Path

# Determine the base directory
BASE_DIR = Path(__file__).resolve().parents[3]
csv_path = BASE_DIR / "data" / "processed" / "foods_processed.csv"

# check null values 
def check_null_values(df):
    list_data_of_food_np_filter = df.dropna()
    return list_data_of_food_np_filter


# Read and process the data
list_data_of_food = pd.read_csv(csv_path)
# drop null values
list_data_of_food = check_null_values(list_data_of_food)
# select specific columns
list_data_of_food_np_filter = list_data_of_food.loc[:, ["dish_name", "calories", "fat", "fiber", "sugar", "protein"]]




calo = 700

# Carb / Sugar (gram)
sugar_min = (calo * 40 / 100) / 4
sugar_max = (calo * 50 / 100) / 4

# Protein (gram)
protein_min = (calo * 15 / 100) / 4
protein_max = (calo * 20 / 100) / 4

# Fat (gram)
fat_min = (calo * 25 / 100) / 9
fat_max = (calo * 30 / 100) / 9

fiber_min = 7



'''
filtered_foods = list_data_of_food_np_filter[
    (list_data_of_food_np_filter["calories"] <= calo) &
    (list_data_of_food_np_filter["sugar"] <= list_data_of_food_np_filter["calories"] * 0.5 / 4) &
    (list_data_of_food_np_filter["fiber"] >= fiber_min) &
    (list_data_of_food_np_filter["fat"] <= list_data_of_food_np_filter["calories"] * 0.5 / 9) &
    (list_data_of_food_np_filter["protein"]<= list_data_of_food_np_filter["calories"] * 0.3 / 4)
] 

print(f"Filtered Foods: {len(filtered_foods)}")
'''
ROLE_KEYWORDS = {
    "carb_main": ["Chay","Cuốn","Spagetty","Ram","BúN","Sandwich", "Ba Chỉ","BúN", "Hoành thánh","cơm", "bún","bánh", "phở", "mì", "bánh mì", "miến", "cháo", "xôi", "khoai", "ngô", "bánh cuốn", "bánh đa", "bánh hỏi", "bánh phở", "bánh tằm", "bánh canh", "bánh gạo", "bánh chưng", "bánh giò", "bánh xèo", "bánh bột lọc", "bánh bèo", "bánh nậm", "bánh ít", "bánh khọt", "bánh đúc", "bánh tét", "bánh chay", "bánh da lợn", "bánh chuối", "bánh khoai", "bánh sắn", "bánh mì", "bánh mì que", "bánh mì trứng", "bánh mì thịt", "bánh mì chả lụa", "bánh bao","tàu hũ ky","Bì","Hủ tíu","Hủ Tiếu"],
    "protein": ["Pate","Egg","Gà","Ghẹ","HếN","XúC XíCh","ThịT","Cá","Xiên","Chả","Xíu Mại", "Cà Ri","LẩU","Bao tử","hải sản","đà điểu","Lẩu","thịt", "cá", "gà", "Bò","bê", "heo", "trứng","Bò","Xúc Xích","Nem","Sushi","hến", "tôm", "đậu", "đậu hũ","mực", "nghêu", "hàu",'sò','nem' "chả", "giò", "lươn", "ốc", "cua","bạch tuộc", "chim", "vịt", "ngỗng", "bò viên", "chả cá", "chả lụa", "chả quế", "chả mực", "chả giò", "chả bò", "chả trứng","ếch", "sườn", "thịt nướng", "thịt kho", "thịt luộc", "thịt xào", "thịt hấp",'nướng', 'hấp', 'luộc', 'xào',"rán","chiên","om","hầm","quay","bít tết","ba rọi","thịt ba chỉ","Chả","Tép","Gan","Lòng","dồi"],
    "vegetable": ["Bí","Rươi","Cà","Dưa ChuộT","Khổ Qua","Bầu","Măng","rau", "canh","Nấm", "salad", "bí", "cải", "su su", "bông", "cà chua", "dưa leo", "rau", "nộm", "gỏi", "canh","xà lách",'lá',"bông","cải","su su","mồng tơi","rau muống","rau dền","rau cải","rau ngót","rau bí","rau lang","rau khoai","rau sam","rau mùng tơi","rau bắp cải","rau súp lơ","bắp cải","súp lơ","cải thảo","cải bó xôi","cải xanh","cải ngọt","dưa chuột","mắm","Súp","bí"],
    "dessert": ["Chanh","Flan","MứT","Chè","Tắc", "Pasta", "Hồng","Sung","Đào", "Smoothie","Vải","Cheesecake","Cà Phê","CóC Ngâm","Cốm","Soda","SấU","Tàu Hũ","SữA Chua","Latte", "BáNh", "Táo","Trà","Milo","Snack","Bacon","Chưng","Detox","Chè","Cookies","Xoài","Chè","Yakult","Nước","MứT" "chè","Trà", "trà", "kem", "trà sữa","sữa", "nước ngọt","Rượu", "sinh tố", "nước ép", "mứt", "kẹo", "bánh ngọt", "bánh kem", "bánh quy", "bánh gato", "bánh su kem", "bánh bông lan", "bánh mì ngọt", "bánh donut", "bánh tart", "bánh mousse", "bánh pudding", "bánh flan","Thạch", "sữa chua", "pudding", "mousse","Bánh Muffin", "mít","siro","chuối","Chocolate", "Cupcake", "Macaron", "Tiramisu", "Brownie","Panna Cotta",'hạt',"Bánh Nhúng","Bánh Phu Thê","Bánh Đậu Xanh","Bánh Tằm Nướng","Bánh Tráng Nướng","Bánh Tráng Trộn","dầm","Sâm","trân châu","trái cây","Cà Phê","Pancake","Waffle","Bingsu","Crepe","Souffle", "Tào Phớ","nha đam"],
}

def classify_dish_multi(dish_name):
    # Bảo vệ dữ liệu bẩn
    if not isinstance(dish_name, str):
        return ["other"]

    # Chuẩn hóa tên món
    name = dish_name.strip().lower()

    # 1️⃣ Dessert override
    for k in ROLE_KEYWORDS.get("dessert", []):
        if k.lower() in name:
            return ["dessert"]

    roles = []

    # 2️⃣ Multi-label cho các nhóm còn lại
    for role, keywords in ROLE_KEYWORDS.items():
        if role == "dessert":
            continue

        for k in keywords:
            if k.lower() in name:
                roles.append(role)
                break  # tránh trùng role nhiều lần

    return roles if roles else ["other"]



list_data_of_food_np_filter['category'] = list_data_of_food_np_filter['dish_name'].apply(classify_dish_multi)
print(list_data_of_food_np_filter["category"].value_counts())
print(list_data_of_food_np_filter[
    ["dish_name", "category"]
].sample(20))
list_data_of_food_np_filter.to_csv(BASE_DIR / "data" / "processed" / "foods_filter.csv", index=False)
'''
list_data_other =  list_data_of_food_np_filter[
    list_data_of_food_np_filter['dish_name'].apply(lambda x: classify_dish_multi(x) == ["other"])
]
print(f"Số món other: {len(list_data_other)}")
print(list_data_other[["dish_name", "calories", "sugar", "fat", "protein", "fiber"]].sample(n=30))
'''