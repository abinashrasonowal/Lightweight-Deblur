from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from torchvision.datasets import STL10

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor()    # Convert the image to a tensor
])

def ApplyBlur(img):
    img_np = np.array(img)

    blurred_img = cv2.GaussianBlur(img_np, (3, 3), 0)

    return blurred_img

class DeblurDataset(Dataset):
    def __init__(self, images, transform=transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        blurred_img = ApplyBlur(img)
        if self.transform:
            img = self.transform(img)
            blurred_img = self.transform(blurred_img)

        return blurred_img, img