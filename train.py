from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import STL10
import torch
import torch.nn as nn
from models import UNet
from loader import DeblurDataset
import torch.optim as optim


stl10_train = STL10(root='./data', split='train', download=True)

images = [img for img, _ in stl10_train]

dataset = DeblurDataset(images)

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

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# Save the model
torch.save(trained_model.state_dict(), 'unet_model_withresnet.pth')
print(f"Model saved")