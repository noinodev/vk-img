import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import io

t1 = pq.read_table("datasets/cifar100/train.parquet")
t2 = pq.read_table("datasets/cifar100/test.parquet")
table = pa.concat_tables([t1, t2])

print(table.schema)
print(table.column_names)

coarse = table["coarse_label"].to_numpy()
fine   = table["fine_label"].to_numpy()
images = table["img"]   # <-- FIXED

count = len(coarse)

# --- inspect first image to determine layout ---
sample_struct = images[0].as_py()
sample_bytes = sample_struct["bytes"]

sample_img = Image.open(io.BytesIO(sample_bytes))
sample_arr = np.array(sample_img, dtype=np.uint8)

if sample_arr.ndim == 3:
    height, width, channels = sample_arr.shape
    image_stride = sample_arr.size
    depth = 1
else:
    raise ValueError("Unsupported image shape")

label_stride = 2
total_stride = label_stride + image_stride

# --- header ---
header = np.zeros(16, dtype=np.uint32)
header[0] = count
header[1] = label_stride
header[2] = image_stride
header[3] = total_stride
header[4] = width
header[5] = height
header[6] = depth
header[7] = channels

with open(input("file output as:"), "wb") as f:
    f.write(header.tobytes())

    for i in range(count):
        f.write(bytes((coarse[i], fine[i])))

        img_struct = images[i].as_py()
        img_bytes = img_struct["bytes"]

        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img, dtype=np.uint8)

        # If you want CIFAR-style planar layout, use transpose:
        # arr = arr.transpose(2, 0, 1)

        f.write(arr.ravel().tobytes())