from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import STL10
from PIL import Image, ImageFilter

class DeblurDataset(Dataset):
    def __init__(self, blurred_images, sharp_images, transform=None):
        self.blurred_images = blurred_images
        self.sharp_images = sharp_images
        self.transform = transform
    
    def __len__(self):
        return len(self.blurred_images)
    
    def __getitem__(self, idx):
        blurred_img = self.blurred_images[idx]
        sharp_img = self.sharp_images[idx]
        
        if self.transform:
            blurred_img = self.transform(blurred_img)
            sharp_img = self.transform(sharp_img)
        
        return blurred_img, sharp_img

# Custom transform to apply blur with kernel size of 3
class ApplyBlur:
    def __init__(self, radius=3):
        self.radius = radius
    
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

# Load STL-10 dataset
stl10_train = STL10(root='./data', split='train', download=True)

# Prepare blurred and sharp images (initially, both are the same)
blurred_images = [img for img, _ in stl10_train]
sharp_images = blurred_images.copy()

# Define the transformation pipeline
transform = transforms.Compose([
    ApplyBlur(radius=3),      # Apply the custom blur
    transforms.ToTensor(),    # Convert the image to a tensor
])

# Create the dataset
deblur_dataset = DeblurDataset(blurred_images, sharp_images, transform=transform)

# Example usage with DataLoader
data_loader = DataLoader(deblur_dataset, batch_size=32, shuffle=True)

# Example to iterate through the DataLoader
for blurred_img, sharp_img in data_loader:
    print(blurred_img.shape, sharp_img.shape)
