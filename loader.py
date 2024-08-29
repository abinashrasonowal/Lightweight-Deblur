from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import albamination as A

# Define the augmentation pipeline
transform2 = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0,value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CenterCrop(height=256, width=256, p=1.0)
])

def augment_both(blurred_img,sharp_img):
  augmented = transform2(image=blurred_img, mask=sharp_img)
  return  augmented['image'] , augmented['mask']

def ApplyBlur(img):
    img_np = np.array(img)
    blurred_img = cv2.GaussianBlur(img_np, (3, 3), 0)
    blurred_img = cv2.GaussianBlur(blurred_img, (3, 3), 0)
    return blurred_img

class MyData(Dataset):
    def __init__(self,df,sharpen_path):
        self.df=df
        self.sp=sharpen_path
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        sharpen_image_path=os.path.join(self.sp,self.df['sharp'][index])

        simg = cv2.imread(sharpen_image_path)
        simg = simg.astype(np.uint8)
        bimg = ApplyBlur(simg)
        bimg = (bimg / 255.0).astype(np.float32)
        simg = (simg / 255.0).astype(np.float32)

        bimg, simg = augment_both(bimg, simg)

        x = torch.tensor(bimg).permute(2, 0, 1)
        y = torch.tensor(simg).permute(2, 0, 1)
        return x,y
