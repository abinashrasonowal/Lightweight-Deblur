from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from models import UNet
from loader import MyData
import torch.optim as optim
import os
import cv2
import pandas as pd

path_to_dir = '.'

img_names = {}
sharp = []

# Define image extensions you want to check for
image_extensions = {'.png', '.jpg', '.jpeg'}

for img in os.listdir(path_to_dir + '/data/sharp'):
    # Get the file extension
    _, ext = os.path.splitext(img)
    # Check if the extension is in the list of image extensions
    if ext.lower() in image_extensions:
        sharp.append(img)

img_names['sharp'] = sharp

df=pd.DataFrame(img_names)

i = 0
invalid_indices = []

for index, image_name in enumerate(df['sharp']):
    path_ = os.path.join(path_to_dir, 'data/sharp', image_name)
    try:
        img = cv2.imread(path_)
        if img is None:
            raise ValueError("Image not loaded, possibly due to unsupported format or missing file.")
    except Exception as e:
        print(f"Error loading image {path_}: {e}")
        invalid_indices.append(index)
        i += 1

# Remove invalid rows from DataFrame
df_cleaned = df.drop(index=invalid_indices).reset_index(drop=True)

print(f"Number of invalid images: {i}")

train_df=df_cleaned[:350]

train_data = MyData(train_df,path_to_dir+'/data/sharp')

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)

model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model


trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)

# Save the model
torch.save(trained_model.state_dict(), 'unet_model_withresnet.pth')
print(f"Model saved")