import os
import sys
import pandas as pd
from sklearn import model_selection

sys.path.append(
    '/home/kento/kaggle_ws/cassava-leaf-disease-classification/')

if 'utils' not in sys.modules:
    from utils.random_seed import SEED


ROOT = '/home/kento/kaggle_ws/cassava-leaf-disease-classification/data/'
SAVE_DIR = '/home/kento/kaggle_ws/cassava-leaf-disease-classification/lists/'
DATA_CSV = os.path.join(ROOT, 'train.csv')
DATA_PATH = os.path.join(ROOT, DATA_CSV)
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = model_selection.train_test_split(
        df, test_size=0.1, random_state=SEED, stratify=df.label.values)
    train_df.to_csv(f'{SAVE_DIR}train.csv', index=False)
    val_df.to_csv(f'{SAVE_DIR}val.csv', index=False)
