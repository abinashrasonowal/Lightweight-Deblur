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
model_path = '/content/drive/MyDrive/DeblurModel/unet_model_withresnet.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

# print(model)
stl10_test = STL10(root='/content/drive/MyDrive/DeblurModel/data', split='test', download=True)

images_test = [img for img, _ in stl10_test]
dataset_test= DeblurDataset(images_test)

input_image = dataset_test[0][1].unsqueeze(0).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
output = model(input_image)

output_image = output.squeeze().cpu().detach().permute(1, 2, 0).numpy()
output_image = (output_image * 255).astype('uint8')

# Save the image
cv2.imwrite("content/drive/MyDrive/DeblurModel/output.png", output_image)
print("output Saved...")

# test_dataloader = DataLoader(images_test, batch_size=16, shuffle=False)

# # Function to test the model
# def test_model(model, dataloader, criterion):
#     model.eval()  # Set the model to evaluation mode
#     running_loss = 0.0
#     with torch.no_grad():  # No need to compute gradients for testing
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             running_loss += loss.item() * inputs.size(0)

#     test_loss = running_loss / len(dataloader.dataset)
#     print(f'Test Loss: {test_loss:.4f}')

# # Define the loss function
# criterion = nn.MSELoss()

# # Test the model
# test_model(model, test_dataloader, criterion)

