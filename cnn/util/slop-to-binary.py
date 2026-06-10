import os
import csv
import numpy as np
from PIL import Image

CLASSES = {
    'n01440764': 0,
    'n02102040': 1,
    'n02979186': 2,
    'n03000684': 3,
    'n03028079': 4,
    'n03394916': 5,
    'n03417042': 6,
    'n03425413': 7,
    'n03445777': 8,
    'n03888257': 9,
}

csv_path = 'datasets/imagenette2-320/noisy_imagenette.csv'
root = 'datasets/imagenette2-320'
SIZE = (224, 224)

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# lock format using first image
img0 = Image.open(os.path.join(root, rows[0]['path'])).convert('RGB')
img0 = img0.resize(SIZE, Image.LANCZOS)

img0_arr = np.array(img0, dtype=np.uint8)
height, width, channels = img0_arr.shape
image_stride = img0_arr.size

label_stride = 2
total_stride = label_stride + image_stride

header = np.zeros(16, dtype=np.uint32)
header[0] = 0
header[1] = label_stride
header[2] = image_stride
header[3] = total_stride
header[4] = width
header[5] = height
header[6] = 1
header[7] = channels

train_out = open('datasets/imagenette2-320/train.bin', 'wb')
val_out   = open('datasets/imagenette2-320/val.bin', 'wb')

train_out.write(header.tobytes())
val_out.write(header.tobytes())

train_count = 0
val_count = 0

for row in rows:
    path = row['path']
    label = CLASSES[row['noisy_labels_0']]
    is_val = row['is_valid'] == 'True'

    img = Image.open(os.path.join(root, path)).convert('RGB')
    img = img.resize(SIZE, Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)

    out = val_out if is_val else train_out

    # 2 byte label
    out.write(np.uint8(label).tobytes())
    out.write(np.uint8(0).tobytes())

    # image
    out.write(arr.ravel().tobytes())

    if is_val:
        val_count += 1
    else:
        train_count += 1

train_out.close()
val_out.close()

print("done")