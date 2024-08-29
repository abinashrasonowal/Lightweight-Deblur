import torch
from models import UNet
from torchvision.datasets import STL10
from models import UNet
from loader import DeblurDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2

model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

