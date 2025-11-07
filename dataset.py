import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import albumentations as albu
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ✅ **只進行 CLAHE 增強對比度**
def apply_CLAHE(image):
    """ 使用 CLAHE 增強對比度 """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result

def preprocess_image(image):
    """ **只執行 CLAHE 增強對比度，不進行去毛髮** """
    return apply_CLAHE(image)

# ✅ **ISIC 2017 訓練集 & 驗證集 Dataset**
class ISIC_2017_Seg_Dataset(Dataset):
    def __init__(self, df, image_size, mode):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode

        if self.mode == 'train':
            self.df = df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.333), interpolation=1, p=1.0),
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.5),
                albu.OneOf([
                    albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
                    albu.GridDistortion(p=1),               
                ], p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.5),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.RandomBrightnessContrast(),
                ], p=0.5),
                albu.HueSaturationValue(p=0.5),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])

        # ✅ **只增強對比度**
        image = preprocess_image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].float().unsqueeze(0)
        mask /= 255.0

        return image, mask

class ISIC_2017_Validation_Dataset(Dataset):
    def __init__(self, csv_path, image_size=512):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size

        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = self.transform(image=image)['image']

        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # 確保 mask 在 0~1 之間

        return image_resized, mask

class ISIC_2017_Seg_Test_Dataset(Dataset):
    def __init__(self, df, image_size=512):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].float().unsqueeze(0) / 255.0  # 確保 mask 在 [0,1] 之間

        return image, mask, row['image_path']
