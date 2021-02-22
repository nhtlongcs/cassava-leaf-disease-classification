import os
import sys
import pandas as pd
from sklearn import model_selection

sys.path.append("./libs")

if "utils" not in sys.modules:
    from utils.random_seed import SEED


ROOT = "./data/"
SAVE_DIR = "./lists/folds/"
DATA_PATH = os.path.join(ROOT, "train.csv")
NUM_FOLDS = 5
if __name__ == "__main__":
    try:
        os.makedirs(SAVE_DIR)
    except:
        pass
    df = pd.read_csv(DATA_PATH)
    skf = model_selection.StratifiedKFold(
        n_splits=NUM_FOLDS, random_state=SEED, shuffle=True
    )
    for idx, fold in enumerate(skf.split(df.image_id.values, df.label.values)):
        train_idxs, val_idxs = fold
        train_df = {
            "image_id": df.image_id.values[train_idxs],
            "label": df.label.values[train_idxs],
        }
        val_df = {
            "image_id": df.image_id.values[val_idxs],
            "label": df.label.values[val_idxs],
        }
        train_df, val_df = (
            pd.DataFrame.from_dict(train_df),
            pd.DataFrame.from_dict(val_df),
        )
        train_df.to_csv(f"{SAVE_DIR}train_{idx}.csv", index=False)
        val_df.to_csv(f"{SAVE_DIR}val_{idx}.csv", index=False)
