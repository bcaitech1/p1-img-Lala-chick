from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd

class MaskDataset(Dataset):
    def __init__(self, df, transforms=None, output_label=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        path = self.df.iloc[idx]['filepath']
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label:
            target = self.df.iloc[idx]['label']
            return img, target
        else:
            return img

class MaskTestDataset(Dataset):
    def __init__(
        self, df, transforms=None, data_root='./input/data/eval/images'
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
          
        path = f"{self.data_root}/{self.df.iloc[index]['ImageID']}"
        
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        return img  