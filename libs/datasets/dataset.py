from torch.utils import data
import torchvision.transforms as tf
import pandas as pd
import os
from PIL import Image
import sys
import cv2

sys.path.append("./libs")

from transforms import *


class LeafDataset(data.Dataset):
    def __init__(
        self, csv_path, data_dir="./data/train_images", IMG_SIZE=256, is_train=True
    ):

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
        # self.data = self.data.loc[:10, :]  # test sample
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


class LeafDatasetAdvance(data.Dataset):
    def __init__(
        self, csv_path, data_dir="./data/train_images", IMG_SIZE=256, is_train=True
    ):

        transforms_train = Compose(
            [
                RandomResizedCrop(IMG_SIZE, IMG_SIZE),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

        transforms_valid = Compose(
            [
                CenterCrop(IMG_SIZE, IMG_SIZE, p=1.0),
                Resize(IMG_SIZE, IMG_SIZE),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

        self.data_dir = data_dir
        self.data = pd.read_csv(csv_path)
        self.data = self.data.loc[:10, :]  # test sample
        self.IMG_SIZE = IMG_SIZE
        self.tf = transforms_train if is_train else transforms_valid
        self.is_train = is_train

    def __getitem__(self, index):
        path, lbl = self.data.values[index]
        path = os.path.join(self.data_dir, path)
        img = cv2.imread(path)[:, :, ::-1]
        img = self.tf(image=img)["image"]
        return img, lbl

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = LeafDataset(
        csv_path="/home/kento/kaggle_ws/cassava-leaf-disease-classification/lists/train.csv",
        data_dir="/home/kento/kaggle_ws/cassava-leaf-disease-classification/data/train_images/",
        IMG_SIZE=256,
        is_train=True,
    )
    val_dataset = LeafDataset(
        csv_path="/home/kento/kaggle_ws/cassava-leaf-disease-classification/lists/val.csv",
        data_dir="/home/kento/kaggle_ws/cassava-leaf-disease-classification/data/train_images/",
        IMG_SIZE=256,
        is_train=False,
    )

    print(train_dataset[0])
    print(val_dataset[0])
