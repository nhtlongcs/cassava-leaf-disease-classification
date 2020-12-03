from torch.utils import data
import torchvision.transforms as tf
import pandas as pd
import os
from PIL import Image


class LeafDataset(data.Dataset):
    def __init__(self,  csv_path, data_dir='./data/train_images', IMG_SIZE=256, is_train=True):

        transforms_train = tf.Compose(
            [
                tf.Resize((IMG_SIZE, IMG_SIZE)),
                tf.RandomHorizontalFlip(p=0.3),
                tf.RandomVerticalFlip(p=0.3),
                tf.RandomResizedCrop(IMG_SIZE),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transforms_valid = tf.Compose(
            [
                tf.Resize((IMG_SIZE, IMG_SIZE)),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_path)
        self.IMG_SIZE = IMG_SIZE
        self.tf = transforms_train if is_train else transforms_valid
        self.is_train = is_train

    def __getitem__(self, index):
        path, lbl = self.data.values[index]
        path = os.path.join(self.data_dir, path)
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        return img, lbl

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = LeafDataset(csv_path='/home/kento/kaggle_ws/cassava-leaf-disease-classification/lists/train.csv',
                                data_dir='/home/kento/kaggle_ws/cassava-leaf-disease-classification/data/train_images/',
                                IMG_SIZE=256, is_train=True)
    val_dataset = LeafDataset(csv_path='/home/kento/kaggle_ws/cassava-leaf-disease-classification/lists/val.csv',
                              data_dir='/home/kento/kaggle_ws/cassava-leaf-disease-classification/data/train_images/',
                              IMG_SIZE=256, is_train=False)

    print(train_dataset[0])
    print(val_dataset[0])
