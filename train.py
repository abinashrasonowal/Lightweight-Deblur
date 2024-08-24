import torch.optim as optim
from models import UNet
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from loader import DeblurDataset

dataset = DeblurDataset()

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
