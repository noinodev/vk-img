import torch
import numpy as np
import os

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_vgg16_bn', pretrained=True)
model.eval()

os.makedirs('cnn/weights/vgg-16', exist_ok=True)

# fold batchnorm into conv weights and biases
features = model.features
i = 0
conv_idx = 0
while i < len(features):
    layer = features[i]
    if isinstance(layer, torch.nn.Conv2d):
        bn = features[i+1]  # batchnorm always follows conv in vgg_bn
        
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(w.shape[0])
        
        # batchnorm params
        gamma = bn.weight.detach().numpy()
        beta  = bn.bias.detach().numpy()
        mean  = bn.running_mean.detach().numpy()
        var   = bn.running_var.detach().numpy()
        eps   = bn.eps
        
        # fold: w' = w * gamma / sqrt(var + eps)
        #        b' = (b - mean) * gamma / sqrt(var + eps) + beta
        scale = gamma / np.sqrt(var + eps)
        w_folded = w * scale[:, None, None, None]
        b_folded = (b - mean) * scale + beta
        
        w_folded.astype(np.float32).tofile(f'cnn/weights/vgg-16/conv{conv_idx}_w.bin')
        b_folded.astype(np.float32).tofile(f'cnn/weights/vgg-16/conv{conv_idx}_b.bin')
        print(f'conv{conv_idx} weights {w_folded.shape} bias {b_folded.shape}')
        
        conv_idx += 1
        i += 2  # skip batchnorm
    else:
        i += 1

# fc layers
for name, param in model.classifier.named_parameters():
    data = param.detach().numpy().astype(np.float32)
    data.tofile(f'cnn/weights/vgg-16/fc_{name}.bin')
    print(f'fc_{name} {data.shape}')