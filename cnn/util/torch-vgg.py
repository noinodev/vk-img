import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(), 
    # match to3d.comp, cifar100 normalization constants
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.CIFAR100(root='./cnn/data', train=True, download=True, transform=transform)

# To match your 500-image overfitting test, create a subset
# Remove these 2 lines if you want to switch back to the full 50,000 images later
subset_indices = list(range(10000))
trainset = torch.utils.data.Subset(trainset, subset_indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

class VulkanMiniVGG7(nn.Module):
    def __init__(self):
        super(VulkanMiniVGG7, self).__init__()
        
        # 5 Convolutional & Pooling Blocks mapping directly to your Lua layout
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 16x16 -> 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 8x8 -> 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 4x4 -> 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 2x2 -> 2x2 -> 1x1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            # First linear layer receives 512 elements from the 1x1x512 feature map
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            # Second linear layer outputs raw logits for the 100 classes
            nn.Linear(512, 100)
            # PyTorch's CrossEntropyLoss calculates Softmax internally, 
            # so adding a manual Softmax layer here is omitted.
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Collapses to a 512-dimension vector
        x = self.classifier(x)
        return x

model = VulkanMiniVGG7().cuda()

# CrossEntropyLoss expects unnormalized logits, perfectly mirroring your manual softmax grad pipeline
criterion = nn.CrossEntropyLoss()

# Switched to Adam to match your custom GLSL shader setup (lr=1e-4, eps=1e-8)
optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

model.train()
epochs = 500 # Adjusted to match your loop length

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].cuda(), data[1].cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track accuracy over the subset to compare with your Lua metrics
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
        
        # Print metrics at the end of the subset epoch
        if i == len(trainloader) - 1:
            print(f'[Epoch {epoch + 1}] Average Loss: {running_loss / len(trainloader):.3f} | Rolling Accuracy: {correct / total * 100:.2f}%')
            running_loss = 0.0