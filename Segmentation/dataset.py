
import pandas as pd

from PIL import Image, ImageOps
import torchvision.transforms as transforms


class BrainMRIDataset:
    def __init__(self, df, transforms,augmentations):
        self.df = df
        self.transforms = transforms
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 1])
        mask = Image.open(self.df.iloc[idx, 2])
        mask = ImageOps.grayscale(mask)
        
        normal = self.transforms()
        augs = self.augmentations()
        image = normal(image)
        image = augs(image)
        mask = normal(mask)
        mask = augs(mask)
        return image,mask

