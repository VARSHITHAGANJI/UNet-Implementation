from dataset import BrainMRIDataset
from transforms import transform_normal,augmentations

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageOps
import random

BASE_PATH =  "lgg-mri-segmentation\kaggle_3m"

data = []

for dir_ in os.listdir(BASE_PATH):
    dir_path = os.path.join(BASE_PATH, dir_)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            data.append([dir_, img_path])
    else:
        print(f"[INFO] This is not a dir --> {dir_path}")
        
df = pd.DataFrame(data, columns=["dir_name", "image_path"])

BASE_LEN = 73
END_LEN = 4
END_MASK_LEN = 9

IMG_SIZE = 512
df_imgs = df[~df["image_path"].str.contains("mask")]
df_masks = df[df["image_path"].str.contains("mask")]


imgs = sorted(df_imgs["image_path"].values, key= lambda x: int(x[BASE_LEN: -END_LEN]))
masks = sorted(df_masks["image_path"].values, key=lambda x: int(x[BASE_LEN: -END_MASK_LEN]))

# idx = random.randint(0, len(imgs)-1)
# idx = 19
# print(f"This image *{imgs[idx]}*\n Belongs to the mask *{masks[idx]}*")

# final dataframe
dff = pd.DataFrame({"patient": df_imgs.dir_name.values,  "image_path": imgs,   "mask_path": masks})

def pos_neg_diagnosis(mask_path):
    val = np.max(Image.open(mask_path))
    if val > 0: return 1
    else: return 0

dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))




train_df, val_df = train_test_split(dff, stratify=dff.diagnosis, test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df, test_df = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.12)
train_df = train_df.reset_index(drop=True)


def train_data():
    train_dataset = BrainMRIDataset(train_df, transform_normal,augmentations)
    train_dataloader = DataLoader(train_dataset, batch_size=26, num_workers=2, shuffle=True)
    return train_dataset,train_dataloader

def valid_data():
    val_dataset = BrainMRIDataset(val_df, transform_normal,augmentations)
    val_dataloader = DataLoader(val_dataset, batch_size=26, num_workers=2, shuffle=True)
    return val_dataset,val_dataloader

def test_data():
    test_dataset = BrainMRIDataset(test_df, transform_normal,augmentations)
    test_dataloader = DataLoader(test_dataset, batch_size=26, num_workers=2, shuffle=True)
    return test_dataset,test_dataloader

