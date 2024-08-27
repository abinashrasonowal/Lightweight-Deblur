import torch
from models import UNet

model = UNet()

model_path = 'unet_model_withresnet.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

