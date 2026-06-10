import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# -------------------------
# Load CIFAR-100 test set
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

# -------------------------
# Load pretrained model
# -------------------------
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar100_vgg16_bn',
    pretrained=True
)

model.eval()

# -------------------------
# Load your exported weights
# -------------------------
def load_exported_weights(model, weight_dir="weights"):
    features = model.features
    conv_idx = 0
    i = 0

    while i < len(features):
        layer = features[i]

        if isinstance(layer, torch.nn.Conv2d):
            bn = features[i + 1]

            w = np.fromfile(f"{weight_dir}/conv{conv_idx}_w.bin", dtype=np.float32)
            b = np.fromfile(f"{weight_dir}/conv{conv_idx}_b.bin", dtype=np.float32)

            w = torch.from_numpy(w).view_as(layer.weight.data)
            b = torch.from_numpy(b).view_as(layer.bias.data)

            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)

            conv_idx += 1
            i += 2
        else:
            i += 1

    # FC layers
    for name, param in model.classifier.named_parameters():
        path = f"{weight_dir}/fc_{name}.bin"
        data = np.fromfile(path, dtype=np.float32)
        data = torch.from_numpy(data).view_as(param.data)
        param.data.copy_(data)

    return model

model = load_exported_weights(model)

# -------------------------
# Evaluation loop
# -------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total:.2f}%")