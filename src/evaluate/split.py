import pandas as pd
from sklearn.model_selection import train_test_split

def split_ratings(path, test_size=0.2):
    df = pd.read_csv(path)

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )

    return train, test